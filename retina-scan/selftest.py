"""
Quick smoke test - run this right after `pip install -r requirements.txt`
to confirm everything is wired up before you demo.

    python selftest.py

It generates a synthetic fundus image and pushes it through the whole
pipeline (preprocess -> quality -> lesions -> inference -> PDF). Works even
if torch is not installed (falls back to the classical CV grader).
"""
import sys
import traceback

OK, FAIL = "PASS", "FAIL"
results = []


def step(name, fn):
    try:
        out = fn()
        results.append((OK, name, ""))
        return out
    except Exception as exc:
        results.append((FAIL, name, f"{exc}"))
        traceback.print_exc()
        return None


def main():
    import numpy as np

    import config
    import preprocessing
    import lesion_analysis
    import dr_model
    import multi_disease
    import ood
    import uncertainty
    import report
    import make_sample

    step("import modules", lambda: True)

    path = step("generate synthetic fundus", lambda: (make_sample.make("sample_fundus.png"), "sample_fundus.png")[1])
    if not path:
        path = "sample_fundus.png"

    rgb = step("load_rgb", lambda: preprocessing.load_rgb(path))
    q = step("quality_check", lambda: preprocessing.quality_check(path))
    step("preprocess (enhanced)", lambda: preprocessing.preprocess(rgb, enhance=True))
    raw = step("preprocess (raw)", lambda: preprocessing.preprocess(rgb, enhance=False))
    lesion = step("lesion_analysis.analyze", lambda: lesion_analysis.analyze(raw))

    # ---- multi-condition screening --------------------------------------
    md = step("multi_disease.screen", lambda: multi_disease.screen(raw))
    if md is not None:
        for f in md["findings"]:
            print(f"    -> {f['condition']}: {f['status']} - {f['headline']}")

    # ---- OOD -------------------------------------------------------------
    ok_ood = step("ood.check (real fundus)", lambda: ood.check(raw))
    if ok_ood is not None:
        print(f"    -> fundus-likelihood {ok_ood['likelihood']} "
              f"({ok_ood['level']}), is_ood={ok_ood['is_ood']}")

    def _junk_is_rejected():
        # A flat grey frame is not a retina. The detector must say so — this is
        # the check that proves abstention actually fires.
        junk = np.full((512, 512, 3), 130, dtype="uint8")
        res = ood.check(junk)
        assert res["is_ood"], f"OOD detector failed to reject a blank frame: {res}"
        return res
    junk = step("ood.check rejects a non-fundus frame", _junk_is_rejected)
    if junk is not None:
        print(f"    -> rejected, likelihood {junk['likelihood']} "
              f"(weakest cue: {junk['weakest_cue']})")

    # ---- uncertainty maths ----------------------------------------------
    def _uncertainty_maths():
        # Two views that disagree must yield low agreement and high entropy.
        views = np.array([[0.05, 0.45, 0.42, 0.05, 0.03],
                          [0.04, 0.40, 0.48, 0.05, 0.03]])
        m = uncertainty.quantify(views)
        assert m["tta_agreement"] < 1.0, "disagreeing views should not agree"
        assert m["margin"] < 0.15, "near-tied classes should give a small margin"
        d = uncertainty.decide(m, quality_score=80)
        assert d["verdict"] in ("borderline", "inconclusive"), d
        # A confident, unanimous prediction must pass.
        sure = np.array([[0.01, 0.02, 0.02, 0.05, 0.90]] * 4)
        d2 = uncertainty.decide(uncertainty.quantify(sure), quality_score=85)
        assert d2["verdict"] == "confident", d2
        # Temperature scaling must soften without changing the argmax.
        p = uncertainty.apply_temperature([0.01, 0.02, 0.02, 0.05, 0.90], 2.0)
        assert int(np.argmax(p)) == 4 and p[4] < 0.90, p
        return m, d, d2
    step("uncertainty: quantify / decide / temperature", _uncertainty_maths)

    def _fail_safe_escalation():
        # Grades 1 and 2 straddle the referral threshold. A coin-flip between
        # them must NOT be reported as "no result" — it must be escalated to the
        # referable grade, because under-referring is the harmful error.
        straddle = np.array([[0.01, 0.50, 0.48, 0.01, 0.00]] * 4)
        d = uncertainty.decide(uncertainty.quantify(straddle), quality_score=85)
        assert d["report_grade"], f"a straddling tie should still report: {d}"
        assert d["escalated"] and d["effective_grade"] == 2, d
        # Grades 3 and 4 are on the same side of the threshold: same referral
        # either way, so the tie is only a note and nothing is escalated.
        within = np.array([[0.0, 0.01, 0.03, 0.48, 0.48]] * 4)
        d2 = uncertainty.decide(uncertainty.quantify(within), quality_score=85)
        assert not d2["escalated"] and d2["report_grade"], d2
        return d, d2
    fs = step("uncertainty: referral-band fail-safe escalation", _fail_safe_escalation)
    if fs is not None:
        print(f"    -> straddling tie -> grade {fs[0]['effective_grade']} "
              f"({fs[0]['verdict']}, escalated={fs[0]['escalated']})")

    step("uncertainty.fit_temperature",
         lambda: uncertainty.fit_temperature(
             np.random.RandomState(0).randn(64, config.NUM_CLASSES) * 3,
             np.random.RandomState(1).randint(0, config.NUM_CLASSES, 64)))

    model = step("load model", lambda: dr_model.get_model())
    if model is not None:
        print(f"    -> model mode: {model.mode}  ({model.source})")

    result = step("run_inference (full pipeline)",
                  lambda: dr_model.run_inference(
                      rgb, enhance=True,
                      quality_score=(q or {}).get("score"),
                      sharpness=(q or {}).get("sharpness")))
    if result is not None:
        u = result["uncertainty"]
        d = result["decision"]
        print(f"    -> grade {result['grade']} ({result['label']}), "
              f"conf {result['confidence']*100:.0f}%, engine={result['mode']}")
        print(f"    -> verdict: {d['verdict']} (report_grade={d['report_grade']})")
        print(f"    -> {uncertainty.summary_line(u)}")
        if result.get("multi_disease"):
            print(f"    -> other conditions: "
                  f"{result['multi_disease']['n_referrals']} referral(s), "
                  f"{result['multi_disease']['n_borderline']} borderline")

        step("run_inference with TTA off",
             lambda: dr_model.run_inference(rgb, enhance=True, use_tta=False))

        step("build PDF report",
             lambda: open("test_report.pdf", "wb").write(
                 report.build_report(result, q, {"id": "TEST", "name": "Synthetic"})))

    print("\n" + "=" * 48)
    n_fail = 0
    for status, name, msg in results:
        # ASCII only: `python selftest.py > out.txt` on Windows redirects stdout
        # through cp1252, which cannot encode check/cross marks and would kill the
        # run with UnicodeEncodeError at the very last step.
        mark = "[ok]" if status == OK else "[!!]"
        print(f"  {mark} {status}  {name}" + (f"   [{msg}]" if msg else ""))
        n_fail += status == FAIL
    print("=" * 48)
    if n_fail:
        print(f"{n_fail} step(s) failed.")
        sys.exit(1)
    print("All good. Now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
