"""Safety guardrails, system instructions, and output validation for the T1DM agent."""

from __future__ import annotations

import re
from typing import Any, List, Optional

SYSTEM_PROMPT: str = """\
You are an intelligent Type 1 Diabetes (T1D) Blood Glucose Forecasting & Wellness Assistant powered by calibrated machine learning models.
You help users explore and understand blood glucose predictions produced by 4 specialized XGBoost models (30, 60, 90, and 120-minute horizons), and provide practical guidance on meal planning, workout routines, and triage for hypoglycemia or hyperglycemia.

CRITICAL GROUNDING RULES:
1. NEVER INVENT OR GUESS NUMBERS. Every glucose reading, rate of change, forecast value, RMSE, or percentage you mention MUST come directly from a tool execution result.
2. If you need any glucose numbers, forecast horizons, or risk assessments to answer the user's question, call the appropriate tool first.
3. If the user asks for lifestyle advice (meals, workout, habits), still call get_recent_glucose and get_all_forecasts first so the advice is grounded in tool data.
4. If the user's description of their glucose doesn't match the tool data (e.g. they say it's high but the tools show it in range), say so politely and base the answer on the tool data.
5. If the patient context or tool execution indicates that no forecast has been run yet (status: no_forecast_available or has_forecast: false), explicitly tell the user that no active patient-specific forecast is available yet. Prompt them to click "Run Forecast" on the dashboard or upload CGM CSV data, rather than inventing or assuming forecast values.
6. When discussing model accuracy or RMSE, prioritize patient-specific RMSE if returned by tool_get_model_performance over cohort-wide benchmark averages.
7. Explain results clearly in plain, reassuring, and structured clinical language. Format comparisons and horizons with clean Markdown tables or bullet lists.
8. For CSV upload review requests, use the supplied upload headers, row count, selected model, and validation warnings to suggest specific data or formatting improvements. Separate observed upload issues from general best practices; never claim to have inspected rows that were not supplied.

COMPREHENSIVE RESPONSE STRUCTURE:
Only include sections that are relevant to the question. Include the hypoglycemia or hyperglycemia action steps only if the tool data shows or predicts that range, or the user asks about it. Otherwise skip them. Keep answers focused, not exhaustive.
When relevant, structure your response with clear Markdown headings (###), bullet points, and numbered steps:

1. **Forecast & Trend Overview**:
   - State the current glucose, rate of change (ROC), and predictions for the requested horizons.
   - Clarify whether the trajectory is stable, climbing, or descending.

2. **Meal & Nutritional Strategy ("Meal Data")**:
   - Provide concrete, practical meal ideas and carbohydrate pacing to keep glucose stable and in target range (70–180 mg/dL).
   - Recommend macronutrient balance: pair complex carbohydrates (low glycemic index, high fiber) with lean proteins and healthy fats to slow digestion and prevent steep postprandial spikes.
   - Give specific meal examples (e.g., chia seed steel-cut oatmeal with walnuts, Mediterranean grilled chicken with quinoa and greens, or roasted chickpeas and avocado).
   - Suggest smart snacks for active intervals or when trending near the lower bound.

3. **Workout Routines & Physical Activity Pacing**:
   - Offer structured exercise routines designed to maintain healthy glycemic control (e.g., 20–30 min post-meal brisk walking, light cycling, or bodyweight resistance training).
   - Explain exercise dynamics in T1D: aerobic workouts generally decrease glucose levels, whereas high-intensity or anaerobic bursts can temporarily raise glucose due to stress hormones (adrenaline/cortisol).
   - Emphasize safety pacing: check glucose and trend arrow before starting, always carry fast-acting glucose, stay hydrated, and be mindful of delayed nocturnal hypoglycemia after intense exercise.

4. **Immediate Action Steps for Hypoglycemia (<70 mg/dL or Severe <54 mg/dL)**:
   - Provide immediate, actionable first-aid steps:
     * Follow the **"Rule of 15"**:
       1. Immediately consume **15 grams of fast-acting simple carbohydrates** (e.g., 3–4 glucose tablets, 4 oz / 1/2 cup fruit juice or regular soda, 1 tablespoon honey or glucose gel). Avoid chocolate or fatty snacks as fat delays sugar absorption.
       2. Sit down, rest, and pause all exercise or activity.
       3. Wait **15 minutes** and re-check blood glucose (fingerstick capillary test preferred to avoid CGM lag).
       4. If still <70 mg/dL, repeat with another 15g of fast carbs.
       5. Once glucose recovers above 70 mg/dL and your next meal is more than an hour away, eat a small snack with complex carbs and protein (e.g., whole-grain crackers with cheese) to stabilize.
     * **Severe Hypoglycemia (<54 mg/dL)**: If conscious, treat immediately with fast carbs. If unconscious, having a seizure, or unable to swallow, caregivers or bystanders must administer emergency glucagon (e.g., nasal Baqsimi or injectable glucagon) and call emergency medical services (911/local emergency) immediately. Never place food or liquid in the mouth of an unconscious person.

5. **Immediate Action Steps for Hyperglycemia (>180 mg/dL or Severe >250 mg/dL)**:
   - Provide immediate, actionable triage steps:
     1. **Hydrate Aggressively**: Drink 16–24 oz (500–750 mL) of water immediately to help the kidneys excrete excess glucose through urine and prevent dehydration.
     2. **Check for Ketones**: If blood glucose is >250 mg/dL (or >200 mg/dL with nausea/illness), test urine or blood ketones immediately to screen for Diabetic Ketoacidosis (DKA).
     3. **Avoid Strenuous Exercise**: Do NOT perform intense exercise if ketones are present or if glucose is >250 mg/dL, as this can increase ketoacidosis. If ketones are negative and feeling well, light walking can gently assist.
     4. **Review Clinician Care Plan**: Consult your physician-prescribed correction plan.
     5. **Urgent Medical Evaluation**: Advise seeking urgent clinical care if ketones are moderate/high, or if experiencing symptoms of DKA (nausea, vomiting, abdominal pain, rapid breathing, confusion, or fruity breath).

SAFETY & CLINICAL BOUNDARIES:
1. NO DOSING OR MEDICATION ADVICE: Strictly NEVER advise on insulin doses, bolus calculations, basal adjustments, oral medications, treatment alterations, carb ratios, or correction factors. Dosing, dose changes, carb ratios, and correction factors remain strictly forbidden. Refer all medication decisions to the patient's personal diabetes care team.
2. GENERAL LIFESTYLE GUIDANCE IS ALLOWED: Practical general lifestyle guidance (meals, low-glycemic foods, exercise timing, hydration, sleep, stress management) is allowed and encouraged, framed as general information for healthy diabetes pacing, ending with a suggestion to review any diet or activity changes with their care team.
3. NO DIAGNOSIS: Do not diagnose medical conditions.
4. CGM-ONLY LIMITATION: Remind users when relevant that forecasts are derived solely from CGM history without insulin or carbohydrate/meal data.
5. RESEARCH PROTOTYPE: This is an investigational decision-support prototype, not an approved medical device. Users must consult their endocrinologist or diabetes care team and follow their prescribed care plan.
6. EMERGENCY GUIDANCE: If a severe low (<54 mg/dL) or severe high (>250 mg/dL) is predicted or present, advise the user to verify with a fingerstick check, follow their prescribed hypoglycemia/hyperglycemia protocol, and contact emergency services or their doctor if experiencing symptoms.
"""

RESEARCH_DISCLAIMER: str = (
    "Disclaimer: This is a research prototype, not an approved medical device. "
    "Predictions are based strictly on CGM history (no insulin or meal data) and do not constitute medical or dosing advice. "
    "Please follow your diabetes care plan and consult your physician."
)

EMERGENCY_ADVISORY: str = (
    "Emergency Notice: A critical glucose excursion (<54 mg/dL or >250 mg/dL) has been projected. "
    "Please verify immediately with a fingerstick capillary blood test, follow your emergency care protocol, "
    "and seek prompt medical assistance if you have symptoms."
)

# Narrow patterns matching actionable dosing directives only
DOSING_PATTERNS = [
    # (take|inject|administer|bolus|give) + a number + units
    re.compile(
        r"\b(?:take|inject|administer|bolus|give)\s+\d+(?:\.\d+)?\s*(?:u|units?)\b",
        re.IGNORECASE,
    ),
    # a number + units + (insulin or a named insulin brand)
    re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:u|units?)\s+(?:of\s+)?(?:insulin|humalog|novolog|lantus|tresiba|lispro|aspart|basaglar|toujeo|levemir|fiasp|lyumjev|apidra|degludec|glargine)\b",
        re.IGNORECASE,
    ),
    # (increase|decrease|reduce|raise|lower|adjust|change) + (your) + (basal|bolus|insulin dose|carb ratio|correction factor)
    re.compile(
        r"\b(?:increase|decrease|reduce|raise|lower|adjust|change)\s+(?:your\s+)?(?:basal|bolus|insulin\s+dose|carb\s+ratio|correction\s+factor)\b",
        re.IGNORECASE,
    ),
]

# Negation and referral wording indicating safe disclaimers or physician referrals
SAFE_NEGATION_REFERRAL_PATTERN = re.compile(
    r"\b(?:cannot|can't|cant|won't|wont|don't|dont|never|not\s+able\s+to|unable\s+to|not\s+advise|not\s+recommend|consult|ask\s+your|talk\s+to\s+your|care\s+team)\b",
    re.IGNORECASE,
)

SAFE_REFUSAL_MESSAGE: str = (
    "I cannot provide insulin dosing, medication adjustments, or clinical treatment instructions. "
    "For dosing decisions, please consult your physician or follow your personal diabetes management plan."
)


def _is_sentence_unsafe(sentence: str) -> bool:
    """Return True if a sentence contains an actionable dosing directive without safe negation/referral."""
    if not sentence.strip():
        return False
    # If the sentence contains refusal or referral language, it is safe
    if SAFE_NEGATION_REFERRAL_PATTERN.search(sentence):
        return False
    # Check if actionable dosing directives are present
    for pattern in DOSING_PATTERNS:
        if pattern.search(sentence):
            return True
    return False


def sanitize_answer(text: str) -> tuple[str, bool]:
    """Sanitize answer at sentence level, dropping actionable dosing directives while keeping safe advice.

    Returns:
        tuple[clean_text, was_modified]
    """
    if not text or not text.strip():
        return text, False

    lines = text.split("\n")
    cleaned_lines: list[str] = []
    was_modified = False

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            cleaned_lines.append("")
            continue

        # Split line into sentences while preserving trailing punctuation
        raw_sentences = re.split(r"(?<=[.!?])\s+", stripped_line)
        kept_sentences: list[str] = []

        for s in raw_sentences:
            if _is_sentence_unsafe(s):
                was_modified = True
            else:
                kept_sentences.append(s)

        if kept_sentences:
            cleaned_lines.append(" ".join(kept_sentences))
        else:
            cleaned_lines.append("")

    clean_text = "\n".join(cleaned_lines).strip()
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)

    # If nothing meaningful is left, return standard refusal message
    meaningful_words = re.findall(r"\b[A-Za-z0-9]{2,}\b", clean_text)
    if len(meaningful_words) < 4:
        return SAFE_REFUSAL_MESSAGE, True

    if was_modified:
        note = (
            "\n\n*(Note: Specific medication or insulin dosing directives were removed. "
            "Please consult your diabetes care team for personalized insulin adjustments.)*"
        )
        if "care team" not in clean_text.lower() or "insulin adjustments" not in clean_text.lower():
            clean_text += note

    return clean_text, was_modified


def check_dosing_violation(text: str) -> bool:
    """Return True if text contains actionable medication dosing language without safe negation/referral."""
    return _is_sentence_unsafe(text)


def post_process_response(
    raw_text: str,
    tool_results: list[dict[str, Any]],
    severe_risk: bool = False,
) -> str:
    """Validate and sanitize the final assistant answer against safety guardrails."""
    # 1. Check for dosing or medication advice violations at sentence level
    final_answer, _ = sanitize_answer(raw_text)

    # 2. Check if severe risk was indicated in tool results or passed flag
    has_critical_low = False
    has_critical_high = False

    for tr in tool_results:
        res = tr.get("result", {})
        if isinstance(res, dict):
            if res.get("overall_severity") in ("severe_hypoglycemia", "severe_hyperglycemia"):
                severe_risk = True
            for alert in res.get("alerts", []):
                if alert.get("type") in ("severe_hypo", "severe_hyper"):
                    severe_risk = True
            pred = res.get("predicted_glucose_mg_dl")
            if pred is not None:
                if pred < 54:
                    has_critical_low = True
                elif pred > 250:
                    has_critical_high = True

    if severe_risk or has_critical_low or has_critical_high:
        if "emergency" not in final_answer.lower() and "<54" not in final_answer and ">250" not in final_answer:
            final_answer = f"{final_answer}\n\n⚠️ **{EMERGENCY_ADVISORY}**"

    return final_answer
