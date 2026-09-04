"""
Patient-education knowledge base for RetinaScan.

For each of the 5 ICDR grades this module provides three things the app can
show alongside a prediction:

  1. what_it_is  - a plain-language explanation of that stage, written FOR the
                   patient ("you"), avoiding jargon or explaining it simply
  2. treatment   - how the stage is typically treated / managed
  3. diet        - foods to avoid, and eye-friendly foods that help, tailored
                   to the stage (gentle/preventive early, stricter later)

Everything here is *curated, static content* on purpose:
  * the app is fully offline, so there is no cloud LLM to generate text, and
  * medical guidance must be reliable, not improvised by a generative model.

This is general educational information, NOT a personalised treatment plan.
Diabetic retinopathy is driven by diabetes, so healthy eating always targets
steady blood sugar and blood pressure; what changes with severity is how
strict that needs to be, and (at higher grades) how much blood-pressure and
professional dietitian support matter.

`get_education(grade)` is the single entry point the UI and PDF report call.
"""
from __future__ import annotations

import numpy as np

import config

# --------------------------------------------------------------------------
# Per-stage content: explanation, treatment, and stage-specific diet
# --------------------------------------------------------------------------
STAGE_INFO = {
    0: {
        "what_it_is": (
            "Good news - we didn't find any signs of diabetic eye disease in "
            "this photo. Diabetic retinopathy happens when high blood sugar "
            "slowly damages the tiny blood vessels at the back of your eye. "
            "Right now those vessels look healthy. Because diabetes can still "
            "affect your eyes over time, the best thing you can do is keep your "
            "sugar in check and get your eyes photographed once a year."
        ),
        "treatment": [
            "No eye treatment is needed right now - your goal is simply to keep "
            "it this way.",
            "Keep your blood sugar in your target range (your doctor can tell "
            "you your number).",
            "Look after your blood pressure too, stay active most days, and "
            "don't smoke.",
            "Get your eyes screened at least once a year, even if they feel "
            "perfectly fine.",
        ],
        "diet_rationale": (
            "What you eat directly affects your blood sugar, and steady blood "
            "sugar keeps your eyes safe. You don't need a special 'diet' - just "
            "smart, everyday choices."
        ),
        "foods_to_avoid": [
            "Sugary drinks - cola, packaged juices, and sweetened tea or "
            "coffee (these raise your sugar the fastest)",
            "Sweets and desserts - mithai, cakes, biscuits, ice cream (best "
            "kept for rare occasions)",
            "Too much 'white' food - white rice, white bread and maida (swap "
            "for whole-grain versions)",
            "Deep-fried snacks like samosas, pakoras and chips",
            "Going heavy on salt, which is hard on your blood pressure",
        ],
        "foods_to_favor": [
            "Green leafy vegetables like spinach, methi and other saag",
            "Whole grains and millets - oats, brown rice, jowar, bajra, whole "
            "wheat",
            "Dal, beans and chickpeas - filling and slow to raise your sugar",
            "Healthy fats - a little fish, or flaxseed and walnuts",
            "Plenty of vegetables - the more colours on your plate, the better",
            "Whole fruit in small amounts - a guava, an apple, a few berries "
            "(eat it whole, don't juice it)",
        ],
    },
    1: {
        "what_it_is": (
            "We found the earliest, mildest signs of diabetic eye disease - a "
            "few tiny weak spots in the small blood vessels of your retina. "
            "Your sight is almost certainly fine right now. Here's the good "
            "news: at this early stage, keeping your blood sugar under control "
            "can stop it getting worse, and can sometimes even undo these early "
            "changes. Think of this as an early warning, not a crisis."
        ),
        "treatment": [
            "You don't need any eye procedure yet - the most powerful "
            "'treatment' here is getting your diabetes well controlled.",
            "Keeping blood sugar, blood pressure and cholesterol steady can "
            "stop these early changes, and may even reverse them.",
            "Get your eyes checked again in about 6 to 12 months, or whenever "
            "your eye doctor suggests.",
            "Keep up the daily basics: balanced meals, regular movement, and no "
            "smoking.",
        ],
        "diet_rationale": (
            "This is the stage where food really pays off. Tightening up what "
            "you eat now can hold these early changes still - or even turn them "
            "around."
        ),
        "foods_to_avoid": [
            "Sugary drinks of every kind - cola, packaged juices, energy "
            "drinks, sweet lassi",
            "Sweets and desserts - try to make these a rare treat, not a daily "
            "habit",
            "Refined carbs - white rice, white bread, maida, sugary breakfast "
            "cereals",
            "Deep-fried and oily snacks, and anything cooked in reused oil",
            "Very salty foods like pickles and papad, and extra table salt",
        ],
        "foods_to_favor": [
            "Green leafy vegetables like spinach and methi",
            "Whole grains and millets - oats, brown rice, jowar, bajra",
            "Dal, beans and chickpeas for slow, steady energy",
            "A little fish, or flaxseed and walnuts, for healthy fats",
            "Lots of vegetables at every meal",
            "Whole fruit in small amounts (eat it whole, not as juice)",
            "A small handful of unsalted nuts as a snack",
        ],
    },
    2: {
        "what_it_is": (
            "Your eyes show a moderate amount of change - a few more of those "
            "weak spots and tiny bleeds, and some of the small vessels have "
            "started to narrow. Your vision may still feel normal, but this is "
            "a sign your eyes need closer attention. It's important now to see "
            "an eye specialist and to be a bit stricter with your blood sugar "
            "and blood pressure."
        ),
        "treatment": [
            "Please see an eye specialist (ophthalmologist) for a full "
            "check-up.",
            "Keeping your blood sugar and blood pressure well controlled now "
            "really matters - it slows things down.",
            "If the centre of your retina becomes swollen, the specialist may "
            "give eye injections or a gentle laser treatment.",
            "You'll likely need eye checks more often - usually every 3 to 6 "
            "months.",
        ],
        "diet_rationale": (
            "From here, being consistent with food matters more than ever. "
            "Regular, well-timed meals stop your sugar swinging up and down, "
            "and cutting back on salt protects your blood pressure - which "
            "protects your eyes."
        ),
        "foods_to_avoid": [
            "All sugary drinks and packaged juices - even the '100% fruit' "
            "ones",
            "Sweets, desserts and sugary bakery items",
            "Refined carbs - white rice, maida, white bread (keep portions "
            "small, choose whole grains)",
            "Salty and processed foods - pickles, papad, packaged snacks, extra "
            "salt (these push up blood pressure, which harms the eyes)",
            "Deep-fried foods, and fatty or processed meats",
        ],
        "foods_to_favor": [
            "Green leafy vegetables and other non-starchy vegetables at every "
            "meal",
            "Whole grains and millets - oats, brown rice, jowar, bajra - in "
            "measured portions",
            "Dal, beans and chickpeas for protein and fibre",
            "Fish, flaxseed or walnuts for healthy omega-3 fats",
            "Unsweetened curd or low-fat dairy in moderation",
            "A small portion of whole fruit, spread through the day rather than "
            "all at once",
        ],
    },
    3: {
        "what_it_is": (
            "Quite a few of the small blood vessels in your retina are now "
            "blocked, so parts of the retina aren't getting enough oxygen. "
            "Your sight may still seem okay, but this stage is a serious "
            "warning - without action it can move to the most dangerous stage "
            "before long. The encouraging part: treatment together with tight "
            "control can protect your vision, so it's important to act now "
            "rather than wait."
        ),
        "treatment": [
            "See a retina specialist soon - typically within a few days to "
            "about two weeks.",
            "The specialist may start laser treatment or eye injections to "
            "lower the risk of dangerous new vessels forming.",
            "Work closely with your doctor to get your blood sugar, blood "
            "pressure and cholesterol as steady as possible.",
            "Expect close, frequent follow-up visits.",
        ],
        "diet_rationale": (
            "At this stage, steady blood sugar and blood pressure are part of "
            "your treatment. It's well worth asking your doctor to refer you to "
            "a dietitian who can build an eating plan around your meals and "
            "medicines."
        ),
        "foods_to_avoid": [
            "All sugary drinks and juices - it's best to cut these out "
            "completely",
            "Sweets, desserts and sugary snacks",
            "White rice, maida and white bread - keep to small portions of "
            "whole grains instead",
            "Salty and packaged foods - pickles, papad, chips, ready meals "
            "(high salt raises blood pressure and speeds eye damage)",
            "Deep-fried foods, fatty red meat and processed meats",
            "Alcohol, which makes blood sugar harder to control",
        ],
        "foods_to_favor": [
            "Non-starchy vegetables - fill half your plate with them",
            "Small, measured portions of whole grains and millets",
            "Dal, beans and other pulses for steady energy",
            "Fish, flaxseed or walnuts for healthy fats",
            "Unsweetened curd or low-fat dairy in small amounts",
            "A little whole fruit at a time (never as juice)",
        ],
    },
    4: {
        "what_it_is": (
            "This is the most advanced stage. Because parts of your retina have "
            "been starved of oxygen, your eye has started growing fragile new "
            "blood vessels. These can bleed or pull on the retina and cause "
            "sudden loss of sight, so this needs urgent care. Please try not to "
            "panic - treatment can save a great deal of vision - but it is "
            "important to see a specialist right away."
        ),
        "treatment": [
            "See a retina specialist urgently to protect your sight - please "
            "don't delay.",
            "Laser treatment (called PRP) is the main way to calm down the "
            "abnormal new vessels.",
            "Eye injections may be used to treat bleeding or swelling.",
            "If there is a lot of bleeding inside the eye, or the retina is "
            "being pulled, an operation (vitrectomy) may be needed.",
            "Even after treatment, keeping your diabetes and blood pressure "
            "under tight control stays essential.",
        ],
        "diet_rationale": (
            "Alongside your treatment, steady blood sugar and blood pressure "
            "give your eyes the best chance to heal. Try not to skip meals or "
            "medicines, and ask for a dietitian's help to keep everything "
            "consistent."
        ),
        "foods_to_avoid": [
            "All sugary drinks, juices and sweets - cut these out",
            "Refined carbs like white rice, maida and white bread",
            "Salty and processed foods - pickles, papad, packaged and fried "
            "snacks - to keep blood pressure down",
            "Deep-fried foods, and fatty or processed meats",
            "Alcohol",
            "Skipping meals - this makes your blood sugar swing, which is risky "
            "around treatment",
        ],
        "foods_to_favor": [
            "Non-starchy vegetables at every meal",
            "Small, consistent portions of whole grains and millets",
            "Dal, beans and pulses, and other lean protein",
            "Fish, flaxseed or walnuts for healthy fats",
            "Unsweetened curd or low-fat dairy in moderation",
            "A small piece of whole fruit, eaten with a meal",
        ],
    },
}

EDU_DISCLAIMER = (
    "This information is general education about diabetic retinopathy, not a "
    "personalised treatment plan. Always follow the advice of a qualified eye "
    "specialist and your diabetes care team."
)


def get_education(grade: int) -> dict:
    """Return the full education bundle for a predicted grade (0-4)."""
    grade = int(np.clip(int(grade), 0, config.NUM_CLASSES - 1))
    info = STAGE_INFO[grade]
    return {
        "grade": grade,
        "stage": config.CLASS_NAMES[grade],
        "what_it_is": info["what_it_is"],
        "treatment": list(info["treatment"]),
        "foods_to_avoid": list(info["foods_to_avoid"]),
        "foods_to_favor": list(info["foods_to_favor"]),
        "diet_rationale": info["diet_rationale"],
        "disclaimer": EDU_DISCLAIMER,
    }
