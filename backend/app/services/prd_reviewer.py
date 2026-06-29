import re
from dataclasses import dataclass

from app.models import (
    ClassificationLevel,
    CriterionAnalysis,
    EvaluationStatus,
    Finding,
    ReadinessStatus,
    ReviewRequest,
    ReviewResponse,
    TokenPlan,
    TokenRoute,
)


SECTION_PATTERNS = {
    "opportunity": re.compile(r"\b(problem|opportunity|hypothesis|job[s]? to be done|jtbd|user need)\b", re.I),
    "scope": re.compile(r"\b(scope|non[- ]?goals?|out of scope|requirements?|functional)\b", re.I),
    "ux": re.compile(r"\b(user experience|ux|edge cases?|error states?|fallback|accessibility|workflow)\b", re.I),
    "metrics": re.compile(r"\b(metric|success|guardrail|kpi|analytics|experiment|a/b|baseline)\b", re.I),
}

SENSITIVE_TERMS = re.compile(
    r"\b(pricing|price|policy|marketplace|security|permission|auth|payment|billing|fraud|privacy|pii|algorithm|ranking)\b",
    re.I,
)
NET_NEW_TERMS = re.compile(r"\b(new|launch|net-new|greenfield|capability|platform|integration|workflow)\b", re.I)
MODERATE_TERMS = re.compile(r"\b(migration|incremental|internal tool|backward compatible|workflow update|automation)\b", re.I)
LIGHT_TERMS = re.compile(r"\b(copy|label|tooltip|minor|discoverability|ux parity|visual|layout|button)\b", re.I)


@dataclass(frozen=True)
class DimensionRule:
    key: str
    criteria: str
    missing_summary: str
    section_name: str
    issue: str
    rationale: str
    replacement_text: str


DIMENSION_RULES = [
    DimensionRule(
        key="opportunity",
        criteria="Opportunity & Hypothesis",
        missing_summary="Problem evidence, target user segment, or falsifiable hypothesis is not explicit.",
        section_name="Opportunity & Hypothesis",
        issue="Unsupported or weakly evidenced product hypothesis",
        rationale="Engineering and GTM teams cannot judge priority or implementation tradeoffs without a concrete user problem, baseline, and expected behavior change.",
        replacement_text=(
            "This feature targets [specific user segment] who currently experience [observable problem]. "
            "Baseline evidence: [metric, research note, ticket volume, or experiment result]. "
            "Hypothesis: if we deliver [capability], then [user behavior/business metric] will improve from [baseline] to [target] within [time window]."
        ),
    ),
    DimensionRule(
        key="scope",
        criteria="Product Scope",
        missing_summary="Boundaries, non-goals, or engineer-ready functional requirements are underspecified.",
        section_name="Scope / Non-Goals",
        issue="Scope boundary is not implementation-ready",
        rationale="Ambiguous scope creates rework, hidden dependencies, and inconsistent engineering task breakdowns.",
        replacement_text=(
            "In scope: [capability 1], [capability 2], and [required integration]. "
            "Out of scope: [explicit non-goal], [future enhancement], and [manual process not changed]. "
            "Engineering may consider the PRD complete when each in-scope behavior has acceptance criteria and owner-approved edge-case handling."
        ),
    ),
    DimensionRule(
        key="ux",
        criteria="UX & Impact",
        missing_summary="Edge cases, error states, or adjacent workflow impacts are not covered.",
        section_name="User Experience & Adjacent Impact",
        issue="Adjacent system or second-order UX impact is not examined",
        rationale="A feature can satisfy the happy path while breaking downstream workflows, operations, support, or user trust in adjacent journeys.",
        replacement_text=(
            "Adjacent impact review: this change touches [workflow/system/team]. "
            "Expected downstream effects are [effect 1] and [effect 2]. "
            "Edge cases: [empty state], [permission failure], [timeout/error], and [rollback path]. "
            "Owners for impacted workflows must approve before launch."
        ),
    ),
    DimensionRule(
        key="metrics",
        criteria="Metrics & Data",
        missing_summary="Success metrics, guardrails, instrumentation, or launch thresholds are missing.",
        section_name="Metrics & Data Rigor",
        issue="Metrics plan lacks measurable success and guardrails",
        rationale="Without quantitative success, guardrails, and event instrumentation, launch readiness cannot be evaluated objectively after release.",
        replacement_text=(
            "Success metric: [primary metric] moves from [baseline] to [target] by [date/window]. "
            "Guardrails: [latency/error/support metric] must not degrade by more than [threshold]. "
            "Instrumentation: track [event names/properties] with dashboards owned by [owner]. "
            "Launch decision: proceed, iterate, or rollback based on these thresholds."
        ),
    ),
]


def review_prd(request: ReviewRequest) -> ReviewResponse:
    text = _normalize(request.prd_text)
    classification_text = _normalize(f"{request.feature_name} {request.platform} {request.prd_text}")
    classification, reason = _classify(classification_text, request.attachments, request.config.extra_sensitive_terms)
    analyses, findings = _evaluate_dimensions(text)
    findings = _add_blind_spot_findings(text, findings, request)
    findings = findings[: request.config.max_findings]

    needs_review = sum(1 for item in analyses if item.evaluation == EvaluationStatus.NEEDS_REVIEW)
    if needs_review == 0:
        status = ReadinessStatus.READY
    elif needs_review <= 2 and classification != ClassificationLevel.FULL_WITH_SCRUTINY:
        status = ReadinessStatus.READY_WITH_CAVEATS
    else:
        status = ReadinessStatus.NOT_READY

    critical_blocker = _critical_blocker(findings, analyses)
    token_plan = _token_plan(request, classification)
    critical_actions = [f"Fix: {finding.issue}" for finding in findings[:2]] or ["No critical blocker identified."]
    optimizations = _optimizations(text, token_plan)
    score = max(0, 100 - (needs_review * 18) - (len(findings) * 5))

    return ReviewResponse(
        feature_name=request.feature_name or "Untitled PRD",
        classification_level=classification,
        classification_reason=reason,
        overall_assessment=status,
        dimensional_analysis=analyses,
        critical_blocker=critical_blocker,
        detailed_findings=findings,
        critical_actions=critical_actions,
        optimizations=optimizations,
        token_plan=token_plan,
        score=score,
    )


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _classify(text: str, attachments: list[str], extra_sensitive_terms: list[str]) -> tuple[ClassificationLevel, str]:
    attachment_hint = " Attachments increase review depth." if attachments else ""
    has_extra_sensitive_term = any(term.strip() and re.search(rf"\b{re.escape(term.strip())}\b", text, re.I) for term in extra_sensitive_terms)
    if SENSITIVE_TERMS.search(text) or has_extra_sensitive_term:
        return (
            ClassificationLevel.FULL_WITH_SCRUTINY,
            "Sensitive policy, pricing, marketplace, security, privacy, or payment language detected." + attachment_hint,
        )
    if NET_NEW_TERMS.search(text):
        return ClassificationLevel.FULL, "Net-new capability or larger value stream detected." + attachment_hint
    if MODERATE_TERMS.search(text):
        return ClassificationLevel.MODERATE, "Incremental workflow, migration, or internal process change detected." + attachment_hint
    if LIGHT_TERMS.search(text) and len(text) < 1800:
        return ClassificationLevel.LIGHTER, "Minor UX, copy, parity, or discoverability update detected." + attachment_hint
    return ClassificationLevel.MODERATE, "Defaulting to moderate review because impact is not clearly minor or net-new." + attachment_hint


def _evaluate_dimensions(text: str) -> tuple[list[CriterionAnalysis], list[Finding]]:
    analyses: list[CriterionAnalysis] = []
    findings: list[Finding] = []
    for rule in DIMENSION_RULES:
        has_signal = bool(SECTION_PATTERNS[rule.key].search(text))
        if has_signal:
            analyses.append(
                CriterionAnalysis(
                    criteria=rule.criteria,
                    evaluation=EvaluationStatus.LOOKS_GOOD,
                    problem_summary="Required topic is present; verify the details are backed by evidence and owner approval.",
                )
            )
        else:
            analyses.append(
                CriterionAnalysis(
                    criteria=rule.criteria,
                    evaluation=EvaluationStatus.NEEDS_REVIEW,
                    problem_summary=rule.missing_summary,
                )
            )
            findings.append(
                Finding(
                    issue=rule.issue,
                    rationale=rule.rationale,
                    section_name=rule.section_name,
                    write_ready_replacement_text=rule.replacement_text,
                    severity="Critical" if rule.key in {"scope", "metrics"} else "High",
                )
            )
    return analyses, findings


def _add_blind_spot_findings(text: str, findings: list[Finding], request: ReviewRequest) -> list[Finding]:
    lowered = text.lower()
    has_growth_claim = re.search(r"\b(\d+%|x\b|increase|growth|conversion|revenue|retention)\b", lowered)
    has_baseline = re.search(r"\b(baseline|current|historical|experiment|cohort|sample|confidence)\b", lowered)
    if request.config.check_unsupported_headroom and has_growth_claim and not has_baseline:
        findings.insert(
            0,
            Finding(
                issue="Unsupported headroom assumption",
                rationale="A quantified upside claim without baseline or historical evidence can bias prioritization and launch expectations.",
                section_name="Opportunity & Hypothesis",
                write_ready_replacement_text=(
                    "Headroom assumption: current baseline is [baseline metric] from [source/date]. "
                    "Expected lift is [target] because [evidence]. If measured lift is below [threshold], the team will [iterate/rollback/stop]."
                ),
                severity="Critical",
            ),
        )
    if request.config.check_revisited_hypothesis and "previous" not in lowered and "prior" not in lowered and "experiment" not in lowered:
        findings.append(
            Finding(
                issue="Revisited hypothesis risk not checked",
                rationale="The PRD does not say whether similar ideas or experiments were attempted before, which risks repeating known failures.",
                section_name="Prior Learnings",
                write_ready_replacement_text=(
                    "Prior learnings: similar attempts reviewed include [experiment/doc/link]. "
                    "What changed since then: [new user behavior, platform capability, market condition, or operational constraint]."
                ),
                severity="Medium",
            )
        )
    return findings


def _critical_blocker(findings: list[Finding], analyses: list[CriterionAnalysis]) -> str:
    critical = next((finding for finding in findings if finding.severity == "Critical"), None)
    if critical:
        return critical.issue
    needs = next((item for item in analyses if item.evaluation == EvaluationStatus.NEEDS_REVIEW), None)
    if needs:
        return needs.problem_summary
    return "No launch-blocking gap detected; keep output compact and proceed to human review."


def _token_plan(request: ReviewRequest, classification: ClassificationLevel) -> TokenPlan:
    input_tokens = max(1, round((len(request.prd_text) + len(request.organizational_context or "")) / 4))
    if classification == ClassificationLevel.LIGHTER:
        route = TokenRoute.COMPACT
        model_hint = "Use low-cost classifier/reviewer for minor changes."
        output_target = 650
    elif classification == ClassificationLevel.FULL_WITH_SCRUTINY:
        route = TokenRoute.ADVANCED
        model_hint = "Escalate to strongest model or specialized reviewer only for sensitive scrutiny."
        output_target = 1100
    else:
        route = TokenRoute.STANDARD
        model_hint = "Use standard reviewer model; reserve advanced model for unresolved blockers."
        output_target = 900

    total_estimate = input_tokens + output_target
    budget_status = "Within budget" if total_estimate <= request.token_budget else "Over budget: compact context before LLM review"
    compaction_actions = [
        "Do not repeat original PRD in output.",
        "Limit findings to top 3 gaps.",
        "Cache reusable company standards and metric definitions.",
    ]
    if total_estimate > request.token_budget:
        compaction_actions.insert(0, "Summarize organizational context to decisions, metrics, and prior experiments only.")

    return TokenPlan(
        input_tokens_estimate=input_tokens,
        output_tokens_target=output_target,
        route=route,
        model_hint=model_hint,
        budget_status=budget_status,
        compaction_actions=compaction_actions,
    )


def _optimizations(text: str, token_plan: TokenPlan) -> list[str]:
    actions = ["Add links/citations for any metric, experiment, architecture dependency, or policy claim."]
    if token_plan.route == TokenRoute.ADVANCED:
        actions.append("Run a specialized security/legal/marketplace review before cross-functional signoff.")
    if len(text) > 6000:
        actions.append("Split large PRD into overview, requirements, metrics, and rollout appendices for cheaper review passes.")
    return actions
