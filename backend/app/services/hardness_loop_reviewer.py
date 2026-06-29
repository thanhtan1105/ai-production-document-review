import json
from dataclasses import dataclass

import httpx

from app.config import get_llm_settings
from app.models import AgentStep, EvaluationStatus, ReviewRequest, ReviewResponse
from app.services.prd_reviewer import review_prd


HARDNESS_THRESHOLD = 45
MAX_HARDNESS_ITERATIONS = 4


@dataclass
class HardnessState:
    request: ReviewRequest
    scorecard: ReviewResponse
    hardness_score: int
    reasons: list[str]
    trace: list[AgentStep]


def review_with_hardness_loop(request: ReviewRequest) -> ReviewResponse:
    """Evaluate a PRD with an agent hardness loop.

    The loop is intentionally tool-shaped:
    1. classify task hardness
    2. inspect KB/context coverage
    3. run targeted dimension review
    4. synthesize the final scorecard

    When LangChain + provider config are available, a ReAct-style LangChain
    agent can run those tools. If it fails or no LLM is configured, the same
    tool loop runs deterministically so evaluation remains available locally.
    """
    settings = get_llm_settings()
    if settings.enabled and settings.provider_name.lower() == "ollama":
        try:
            return _run_ollama_tool_agent(request)
        except Exception:
            pass

    if _can_use_langchain_agent():
        try:
            return _run_langchain_react_agent(request)
        except Exception:
            pass
    return _run_deterministic_hardness_loop(request)


def _run_deterministic_hardness_loop(request: ReviewRequest) -> ReviewResponse:
    scorecard = review_prd(request)
    hardness_score, reasons = _classify_hardness(request, scorecard)
    trace = [
        AgentStep(
            step="1",
            tool="classify_review_hardness",
            observation="; ".join(reasons) or "Low-risk PRD with enough visible structure.",
            hardness_score=hardness_score,
        )
    ]
    state = HardnessState(request=request, scorecard=scorecard, hardness_score=hardness_score, reasons=reasons, trace=trace)

    for iteration in range(2, MAX_HARDNESS_ITERATIONS + 1):
        if state.hardness_score < HARDNESS_THRESHOLD:
            state.trace.append(
                AgentStep(
                    step=str(iteration),
                    tool="stop_condition",
                    observation="Hardness below threshold; no further tool calls required.",
                    hardness_score=state.hardness_score,
                )
            )
            break

        if iteration == 2:
            _inspect_knowledge_base(state, iteration)
        elif iteration == 3:
            _run_dimension_review(state, iteration)
        else:
            _synthesize_scorecard(state, iteration)
            break

    return state.scorecard.model_copy(
        update={
            "agent_architecture": "hardness-loop-deterministic",
            "agent_trace": state.trace,
        }
    )


def _inspect_knowledge_base(state: HardnessState, iteration: int) -> None:
    context = state.request.organizational_context or ""
    signals = []
    if len(context) < 500:
        signals.append("Knowledge base context is thin.")
        state.hardness_score += 10
    if "prior" not in context.lower() and "experiment" not in context.lower():
        signals.append("No prior experiment signal found in OpenKB context.")
        state.hardness_score += 8
    if "guardrail" in context.lower() or "metric" in context.lower():
        signals.append("Metric/guardrail standards found in OpenKB context.")
        state.hardness_score = max(0, state.hardness_score - 5)
    if not signals:
        signals.append("OpenKB context has enough broad signals for first-pass review.")

    state.trace.append(
        AgentStep(
            step=str(iteration),
            tool="inspect_openkb_context",
            observation=" ".join(signals),
            hardness_score=state.hardness_score,
        )
    )


def _run_dimension_review(state: HardnessState, iteration: int) -> None:
    needs_review = [
        item.criteria
        for item in state.scorecard.dimensional_analysis
        if item.evaluation == EvaluationStatus.NEEDS_REVIEW
    ]
    if needs_review:
        observation = f"Targeted dimension agents escalated: {', '.join(needs_review)}."
        state.hardness_score += min(12, len(needs_review) * 4)
    else:
        observation = "All dimension reviewers passed; reducing hardness."
        state.hardness_score = max(0, state.hardness_score - 15)
    state.trace.append(
        AgentStep(
            step=str(iteration),
            tool="run_dimension_reviewers",
            observation=observation,
            hardness_score=state.hardness_score,
        )
    )


def _synthesize_scorecard(state: HardnessState, iteration: int) -> None:
    if state.hardness_score >= 70 and "Use standard reviewer model" in state.scorecard.token_plan.model_hint:
        token_plan = state.scorecard.token_plan.model_copy(
            update={
                "model_hint": "Hardness loop recommends advanced reviewer or specialized human review.",
                "compaction_actions": [
                    *state.scorecard.token_plan.compaction_actions,
                    "Escalate unresolved hard dimensions before launch readiness signoff.",
                ],
            }
        )
        state.scorecard = state.scorecard.model_copy(update={"token_plan": token_plan})

    state.trace.append(
        AgentStep(
            step=str(iteration),
            tool="synthesize_scorecard",
            observation="Synthesized final scorecard from hardness classification, OpenKB inspection, and dimension reviewers.",
            hardness_score=state.hardness_score,
        )
    )


def _classify_hardness(request: ReviewRequest, scorecard: ReviewResponse) -> tuple[int, list[str]]:
    score = 10
    reasons: list[str] = []
    if scorecard.classification_level == "Full Review with Specialized Scrutiny":
        score += 35
        reasons.append("Sensitive/specialized scrutiny classification.")
    elif scorecard.classification_level == "Full Review":
        score += 22
        reasons.append("Full review classification.")
    elif scorecard.classification_level == "Moderate Review":
        score += 12
        reasons.append("Moderate change classification.")

    needs_review = sum(1 for item in scorecard.dimensional_analysis if item.evaluation == EvaluationStatus.NEEDS_REVIEW)
    if needs_review:
        score += needs_review * 12
        reasons.append(f"{needs_review} readiness dimension(s) need review.")
    if scorecard.detailed_findings:
        score += min(20, len(scorecard.detailed_findings) * 6)
        reasons.append(f"{len(scorecard.detailed_findings)} detailed finding(s) detected.")
    if "Over budget" in scorecard.token_plan.budget_status:
        score += 10
        reasons.append("Token budget pressure detected.")
    if len(request.prd_text) > 6000:
        score += 8
        reasons.append("Large PRD body.")
    return min(100, score), reasons


def _can_use_langchain_agent() -> bool:
    settings = get_llm_settings()
    if not settings.enabled:
        return False
    if settings.provider_name.lower() == "ollama":
        return False
    try:
        import langchain  # noqa: F401
        import langchain_openai  # noqa: F401
    except Exception:
        return False
    return True


def _run_ollama_tool_agent(request: ReviewRequest) -> ReviewResponse:
    """Run an Ollama-native ReAct-style tool loop.

    Ollama Cloud exposes the same `/api/chat` tool-calling interface as local
    Ollama. This keeps the actual agent loop provider-native while preserving
    the deterministic reviewer as the executable tool implementation.
    """
    settings = get_llm_settings()
    assert settings.base_url and settings.model

    state: HardnessState | None = None
    last_scorecard: ReviewResponse | None = None

    def classify_review_hardness() -> str:
        nonlocal state
        scorecard = review_prd(request)
        hardness_score, reasons = _classify_hardness(request, scorecard)
        trace = [
            AgentStep(
                step="1",
                tool="classify_review_hardness",
                observation="; ".join(reasons) or "Low-risk PRD with enough visible structure.",
                hardness_score=hardness_score,
            )
        ]
        state = HardnessState(request=request, scorecard=scorecard, hardness_score=hardness_score, reasons=reasons, trace=trace)
        return json.dumps(
            {"hardness_score": hardness_score, "reasons": reasons, "scorecard": scorecard.model_dump(mode="json")},
            ensure_ascii=False,
        )

    def inspect_openkb_context() -> str:
        if state is None:
            return json.dumps({"error": "classify_review_hardness must run first"})
        _inspect_knowledge_base(state, 2)
        context = state.request.organizational_context or ""
        return json.dumps(
            {
                "characters": len(context),
                "hardness_score": state.hardness_score,
                "observation": state.trace[-1].observation,
            },
            ensure_ascii=False,
        )

    def run_dimension_reviewers() -> str:
        if state is None:
            return json.dumps({"error": "classify_review_hardness must run first"})
        _run_dimension_review(state, 3)
        return json.dumps(
            {"hardness_score": state.hardness_score, "observation": state.trace[-1].observation},
            ensure_ascii=False,
        )

    def synthesize_scorecard() -> str:
        nonlocal last_scorecard
        if state is None:
            return json.dumps({"error": "classify_review_hardness must run first"})
        if not any(step.tool == "inspect_openkb_context" for step in state.trace):
            _inspect_knowledge_base(state, 2)
        if not any(step.tool == "run_dimension_reviewers" for step in state.trace):
            _run_dimension_review(state, 3)
        _synthesize_scorecard(state, 4)
        last_scorecard = state.scorecard.model_copy(
            update={
                "agent_architecture": "ollama-native-tool-agent",
                "agent_trace": state.trace,
            }
        )
        return last_scorecard.model_dump_json()

    available_tools = {
        "classify_review_hardness": classify_review_hardness,
        "inspect_openkb_context": inspect_openkb_context,
        "run_dimension_reviewers": run_dimension_reviewers,
        "synthesize_scorecard": synthesize_scorecard,
    }
    tool_schemas = [
        {
            "type": "function",
            "function": {
                "name": "classify_review_hardness",
                "description": "Classify the PRD review hardness using launch-readiness risk signals.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "inspect_openkb_context",
                "description": "Inspect product knowledge-base coverage for prior experiments and metric standards.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_dimension_reviewers",
                "description": "Run targeted dimension reviewers and update the hardness score.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "synthesize_scorecard",
                "description": "Synthesize and return the final strict PRD review scorecard JSON.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ]
    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are a PRD review ReAct agent. You are inside a tool loop. "
                "Call tools in this order: classify_review_hardness, inspect_openkb_context, "
                "run_dimension_reviewers, synthesize_scorecard. "
                "After synthesize_scorecard returns JSON, return that JSON only."
            ),
        },
        {"role": "user", "content": "Run the PRD hardness loop and return the final scorecard JSON."},
    ]
    headers = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"

    with httpx.Client(timeout=90) as client:
        for _ in range(MAX_HARDNESS_ITERATIONS + 2):
            response = client.post(
                settings.base_url.rstrip("/") + "/api/chat",
                headers=headers,
                json={
                    "model": settings.model,
                    "messages": messages,
                    "stream": False,
                    "tools": tool_schemas,
                    "options": {"temperature": 0.1},
                },
            )
            response.raise_for_status()
            message = response.json().get("message", {})
            tool_calls = message.get("tool_calls") or []
            messages.append(message)

            if not tool_calls:
                content = message.get("content") or ""
                if content:
                    parsed = ReviewResponse.model_validate_json(content)
                    return parsed.model_copy(
                        update={
                            "agent_architecture": "ollama-native-tool-agent",
                            "agent_trace": parsed.agent_trace,
                        }
                    )
                break

            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                tool_name = function.get("name")
                tool_fn = available_tools.get(tool_name)
                if not tool_fn:
                    tool_result = json.dumps({"error": f"Unknown tool: {tool_name}"})
                else:
                    tool_result = tool_fn()
                messages.append({"role": "tool", "tool_name": tool_name, "content": tool_result})

            if last_scorecard is not None:
                return last_scorecard.model_copy(
                    update={
                        "agent_architecture": "ollama-native-tool-agent",
                        "agent_trace": last_scorecard.agent_trace,
                    }
                )

    if last_scorecard is None:
        raise RuntimeError("Ollama tool agent did not produce a scorecard.")
    return last_scorecard.model_copy(
        update={
            "agent_architecture": "ollama-native-tool-agent",
            "agent_trace": last_scorecard.agent_trace,
        }
    )


def _run_langchain_react_agent(request: ReviewRequest) -> ReviewResponse:
    """Run a LangChain v1 tool-calling agent loop.

    LangChain v1's `create_agent` is the maintained agent harness: model calls
    tools in a loop until the task is complete. We keep this optional because
    local dev should still work without provider keys.
    """
    from langchain.agents import create_agent
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI

    settings = get_llm_settings()
    assert settings.base_url and settings.api_key

    @tool
    def classify_review_hardness() -> str:
        """Classify PRD hardness from the current scorecard."""
        scorecard = review_prd(request)
        hardness_score, reasons = _classify_hardness(request, scorecard)
        return json.dumps({"hardness_score": hardness_score, "reasons": reasons, "scorecard": scorecard.model_dump(mode="json")})

    @tool
    def inspect_openkb_context() -> str:
        """Inspect product knowledge-base context coverage."""
        context = request.organizational_context or ""
        return json.dumps(
            {
                "characters": len(context),
                "has_prior_experiment": "prior" in context.lower() or "experiment" in context.lower(),
                "has_metrics": "metric" in context.lower() or "guardrail" in context.lower(),
            }
        )

    @tool
    def synthesize_scorecard() -> str:
        """Return the final strict JSON scorecard."""
        scorecard = _run_deterministic_hardness_loop(request)
        return scorecard.model_dump_json()

    model = ChatOpenAI(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=0.1,
        timeout=60,
    )
    agent = create_agent(
        model=model,
        tools=[classify_review_hardness, inspect_openkb_context, synthesize_scorecard],
        system_prompt=(
            "You are a PRD review ReAct agent. Use tools in this order unless observations prove it unnecessary: "
            "classify_review_hardness, inspect_openkb_context, synthesize_scorecard. "
            "Return only the JSON emitted by synthesize_scorecard."
        ),
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "Run the PRD hardness loop and return the final scorecard JSON."}]})
    final_message = result["messages"][-1].content
    if isinstance(final_message, list):
        final_message = "".join(str(part) for part in final_message)
    response = ReviewResponse.model_validate_json(str(final_message))
    return response.model_copy(
        update={
            "agent_architecture": "langchain-react-hardness-loop",
            "agent_trace": [
                *response.agent_trace,
                AgentStep(
                    step="langchain",
                    tool="create_agent",
                    observation="LangChain ReAct/tool-calling agent completed the hardness loop.",
                    hardness_score=response.agent_trace[-1].hardness_score if response.agent_trace else 0,
                ),
            ],
        }
    )
