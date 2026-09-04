"""
Generate a synthetic fundus-like image so you can exercise the full pipeline
without a dataset (handy for a quick demo / smoke test).

    python make_sample.py          # writes sample_fundus.png

It is NOT a real retina - just enough structure (disc, vessels, a few bright
and dark lesions) to make preprocessing, lesion detection and the UI light up.
"""
import cv2
import numpy as np

SIZE = 700


def make(path="sample_fundus.png", seed=7):
    rng = np.random.default_rng(seed)
    img = np.zeros((SIZE, SIZE, 3), np.uint8)
    c = (SIZE // 2, SIZE // 2)
    R = int(SIZE * 0.47)

    # retina base (reddish), radial falloff
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    dist = np.sqrt((xx - c[0]) ** 2 + (yy - c[1]) ** 2)
    fov = dist <= R
    base = np.zeros((SIZE, SIZE, 3), np.float32)
    falloff = np.clip(1 - dist / R, 0, 1)
    base[..., 0] = 150 * falloff + 40      # R
    base[..., 1] = 60 * falloff + 15       # G
    base[..., 2] = 45 * falloff + 10       # B
    img[fov] = base[fov].astype(np.uint8)

    # optic disc (bright, slightly off-centre)
    od = (int(SIZE * 0.62), int(SIZE * 0.5))
    cv2.circle(img, od, int(SIZE * 0.06), (235, 220, 150), -1)
    img = cv2.GaussianBlur(img, (0, 0), 3)

    # vessels radiating from disc
    for _ in range(9):
        ang = rng.uniform(0, 2 * np.pi)
        x, y = od
        pts = [(x, y)]
        for _ in range(int(rng.integers(12, 22))):
            ang += rng.uniform(-0.25, 0.25)
            step = rng.uniform(10, 20)
            x += int(step * np.cos(ang)); y += int(step * np.sin(ang))
            pts.append((x, y))
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], (110, 25, 20),
                     thickness=int(rng.integers(1, 4)))

    # exudates (bright yellow spots)
    for _ in range(6):
        a = rng.uniform(0, 2 * np.pi); r = rng.uniform(0.15, 0.42) * R
        p = (int(c[0] + r * np.cos(a)), int(c[1] + r * np.sin(a)))
        cv2.circle(img, p, int(rng.integers(3, 7)), (245, 230, 90), -1)

    # haemorrhages / micro-aneurysms (small dark-red round spots)
    for _ in range(14):
        a = rng.uniform(0, 2 * np.pi); r = rng.uniform(0.15, 0.45) * R
        p = (int(c[0] + r * np.cos(a)), int(c[1] + r * np.sin(a)))
        cv2.circle(img, p, int(rng.integers(2, 5)), (70, 12, 12), -1)

    img[~fov] = 0
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"wrote {path}")


if __name__ == "__main__":
    make()
