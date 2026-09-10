"""GenAI layer: turn a customer's SHAP factors into a plain-English risk
explanation and a constrained retention recommendation, using the Gemini
API free tier.

WHY Gemini instead of OpenAI
------------------------------
This project is built on a $0 budget. Google AI Studio issues a free API
key with no credit card and a generous free-tier request quota, which
OpenAI does not offer -- see the README for the two-minute setup.

WHY a guardrail on the recommended action
--------------------------------------------
An LLM asked to "recommend a retention action" in free text will happily
invent plausible-sounding but operationally meaningless suggestions ("send
a personalized care package", "run a win-back campaign with X, Y, Z...").
A business tool that feeds a CRM or a rep's task queue needs the output to
be one of a small number of actions someone can actually execute. So the
model is instructed to choose from a fixed list AND the response is
validated against that list in code after the fact -- prompting alone is
not trusted, because nothing stops the model from ignoring the instruction.
Anything that doesn't match exactly is flagged for manual review instead of
being silently shown to a business user.

WHY temperature 0.2-0.3
--------------------------
This output may inform a real retention action, not brainstorm creative
copy. A low temperature makes the model's action choice and explanation
consistent for the same customer across repeated calls, which matters for
trust and for the guardrail check (a high-temperature model is more likely
to wander outside the allowed vocabulary).
"""

import json
import os
import re

from config import ALLOWED_ACTIONS

TEMPERATURE = 0.25

SYSTEM_INSTRUCTION = (
    "You are a retention analyst assistant for an online retail business. "
    "You are given one customer's churn-risk score and the top statistical "
    "factors driving that score (from a SHAP explanation of a trained "
    "churn model). Respond ONLY with a JSON object with exactly two keys: "
    '"explanation" (a plain-English, 2-sentence explanation of why this '
    "customer is at risk, written for a non-technical retention rep, with "
    "no jargon like 'SHAP' or 'feature') and \"recommended_action\" "
    "(exactly one string, copied verbatim from the allowed list below -- "
    "no other text, no variations, no combinations).\n\n"
    f"Allowed actions: {json.dumps(ALLOWED_ACTIONS)}"
)


def get_api_key():
    """Resolve the Gemini API key from st.secrets first, then the environment.

    st.secrets is checked first so the same code works unmodified once
    deployed to Streamlit Community Cloud (which injects secrets.toml
    values into st.secrets), while the environment-variable fallback keeps
    local development and the CLI scripts working without Streamlit at all.
    """
    try:
        import streamlit as st

        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")


def get_model_name():
    """Same st.secrets-then-env resolution as get_api_key(), for the
    optional GEMINI_MODEL override (Streamlit Cloud secrets aren't exposed
    as OS environment variables, so both lookups are needed)."""
    try:
        import streamlit as st

        if "GEMINI_MODEL" in st.secrets:
            return st.secrets["GEMINI_MODEL"]
    except Exception:
        pass
    return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def _build_user_prompt(customer_id, churn_probability, top_factors):
    factor_lines = "\n".join(
        f"- {f['feature']} = {f['value']:.2f} ({f['direction']}, impact {f['shap_value']:+.3f})"
        for f in top_factors
    )
    return (
        f"Customer ID: {customer_id}\n"
        f"Predicted churn probability: {churn_probability:.0%}\n"
        f"Top risk factors:\n{factor_lines}\n"
    )


def _extract_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def generate_customer_explanation(customer_id, churn_probability, top_factors, model_name=None):
    """Return a dict describing the customer's risk and a guarded action.

    top_factors: list of dicts with keys feature/value/shap_value/direction
    (the output of explain.get_top_factors(...).to_dict("records")).

    Return shape is stable regardless of success/failure so app.py never
    has to special-case exceptions:
        {
            "explanation": str,
            "recommended_action": str,
            "needs_manual_review": bool,
            "error": str | None,
        }
    """
    api_key = get_api_key()
    if not api_key:
        return {
            "explanation": None,
            "recommended_action": None,
            "needs_manual_review": True,
            "error": (
                "No Gemini API key found. Set GEMINI_API_KEY as an environment "
                "variable or in .streamlit/secrets.toml -- see README for the "
                "free aistudio.google.com setup steps."
            ),
        }

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=model_name or get_model_name(),
            system_instruction=SYSTEM_INSTRUCTION,
        )
        prompt = _build_user_prompt(customer_id, churn_probability, top_factors)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=TEMPERATURE,
                response_mime_type="application/json",
            ),
        )
        parsed = _extract_json(response.text)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI, don't crash it
        return {
            "explanation": None,
            "recommended_action": None,
            "needs_manual_review": True,
            "error": f"Gemini call failed: {exc}",
        }

    explanation = parsed.get("explanation")
    action = (parsed.get("recommended_action") or "").strip()

    # The guardrail: exact, case-insensitive match against the fixed action
    # list. Nothing the model says overrides this check.
    matched_action = next((a for a in ALLOWED_ACTIONS if a.lower() == action.lower()), None)

    if not explanation or not matched_action:
        return {
            "explanation": explanation,
            "recommended_action": action or None,
            "needs_manual_review": True,
            "error": (
                f"Model returned an action outside the allowed list: {action!r}"
                if action and not matched_action
                else "Model response was missing required fields."
            ),
        }

    return {
        "explanation": explanation,
        "recommended_action": matched_action,
        "needs_manual_review": False,
        "error": None,
    }
