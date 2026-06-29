import json

import httpx
from pydantic import ValidationError

from app.config import get_llm_settings
from app.models import ReviewRequest, ReviewResponse
from app.services.hardness_loop_reviewer import review_with_hardness_loop
from app.services.prd_reviewer import review_prd


SYSTEM_PROMPT = """You are a First-Pass PRD Evaluator.
Return ONLY valid JSON matching this schema:
{
  "feature_name": string,
  "classification_level": "Lighter Review" | "Moderate Review" | "Full Review" | "Full Review with Specialized Scrutiny",
  "classification_reason": string,
  "overall_assessment": "Ready" | "Ready with Caveats" | "Not Ready",
  "dimensional_analysis": [{"criteria": string, "evaluation": "Looks Good" | "Needs Review", "problem_summary": string}],
  "critical_blocker": string,
  "detailed_findings": [{"issue": string, "rationale": string, "section_name": string, "write_ready_replacement_text": string, "severity": string}],
  "critical_actions": [string],
  "optimizations": [string],
  "token_plan": {
    "input_tokens_estimate": integer,
    "output_tokens_target": integer,
    "route": "compact-classifier" | "standard-reviewer" | "advanced-scrutiny",
    "model_hint": string,
    "budget_status": string,
    "compaction_actions": [string]
  },
  "score": integer
}
Rules:
- Do not repeat the full PRD.
- Limit detailed_findings to the requested max_findings.
- Evaluate Opportunity & Hypothesis, Product Scope, UX & Impact, Metrics & Data.
- Include write-ready replacement text for each finding.
"""


def review_with_optional_llm(request: ReviewRequest) -> ReviewResponse:
    if request.config.review_mode == "agent":
        return review_with_hardness_loop(request)

    if request.config.review_mode == "heuristic":
        return review_prd(request)

    settings = get_llm_settings()
    if not settings.enabled:
        if request.config.review_mode == "llm":
            raise RuntimeError("LLM mode requested but provider config is incomplete.")
        return review_prd(request)

    try:
        if settings.provider_name.lower() == "ollama":
            return _call_ollama_reviewer(request, settings.base_url or "", settings.api_key, settings.model)
        return _call_openai_compatible_reviewer(request, settings.base_url or "", settings.api_key or "", settings.model)
    except Exception:
        if request.config.review_mode == "llm":
            raise
        return review_prd(request)


def _call_openai_compatible_reviewer(request: ReviewRequest, base_url: str, api_key: str, model: str) -> ReviewResponse:
    url = base_url.rstrip("/") + "/chat/completions"
    fallback = review_prd(request)
    user_prompt = {
        "feature_name": request.feature_name,
        "platform": request.platform,
        "token_budget": request.token_budget,
        "config": request.config.model_dump(),
        "organizational_context": request.organizational_context,
        "prd_text": request.prd_text,
        "fallback_scorecard_shape": fallback.model_dump(mode="json"),
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }

    with httpx.Client(timeout=60) as client:
        response = client.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]

    try:
        return ReviewResponse.model_validate_json(content)
    except ValidationError:
        return ReviewResponse.model_validate(json.loads(content))


def _call_ollama_reviewer(request: ReviewRequest, base_url: str, api_key: str | None, model: str) -> ReviewResponse:
    url = base_url.rstrip("/") + "/api/chat"
    fallback = review_prd(request)
    user_prompt = {
        "feature_name": request.feature_name,
        "platform": request.platform,
        "token_budget": request.token_budget,
        "config": request.config.model_dump(),
        "organizational_context": request.organizational_context,
        "prd_text": request.prd_text,
        "fallback_scorecard_shape": fallback.model_dump(mode="json"),
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT + "\nReturn JSON only. No markdown fences."},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    with httpx.Client(timeout=90) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        content = data.get("message", {}).get("content", "")

    try:
        return ReviewResponse.model_validate_json(content)
    except ValidationError:
        return ReviewResponse.model_validate(json.loads(content))
