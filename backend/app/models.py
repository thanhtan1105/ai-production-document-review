from enum import Enum

from pydantic import BaseModel, Field


class ClassificationLevel(str, Enum):
    LIGHTER = "Lighter Review"
    MODERATE = "Moderate Review"
    FULL = "Full Review"
    FULL_WITH_SCRUTINY = "Full Review with Specialized Scrutiny"


class ReadinessStatus(str, Enum):
    READY = "Ready"
    READY_WITH_CAVEATS = "Ready with Caveats"
    NOT_READY = "Not Ready"


class EvaluationStatus(str, Enum):
    LOOKS_GOOD = "Looks Good"
    NEEDS_REVIEW = "Needs Review"


class TokenRoute(str, Enum):
    COMPACT = "compact-classifier"
    STANDARD = "standard-reviewer"
    ADVANCED = "advanced-scrutiny"


class ReviewMode(str, Enum):
    HEURISTIC = "heuristic"
    LLM = "llm"
    HYBRID = "hybrid"
    AGENT = "agent"


class ReviewConfig(BaseModel):
    review_mode: ReviewMode = ReviewMode.HEURISTIC
    max_findings: int = Field(default=3, ge=1, le=10)
    check_unsupported_headroom: bool = True
    check_revisited_hypothesis: bool = True
    extra_sensitive_terms: list[str] = Field(default_factory=list, max_length=30)


class ReviewRequest(BaseModel):
    feature_name: str = Field(default="Untitled PRD", max_length=160)
    platform: str = Field(default="Unspecified", max_length=120)
    prd_text: str = Field(min_length=80)
    attachments: list[str] = Field(default_factory=list)
    organizational_context: str | None = Field(
        default=None,
        description="Optional compact context such as company metrics, prior experiment summaries, or architecture notes.",
    )
    token_budget: int = Field(default=12000, ge=1000, le=200000)
    config: ReviewConfig = Field(default_factory=ReviewConfig)


class DocumentExtractResponse(BaseModel):
    filename: str
    text: str
    character_count: int


class RuntimeConfigResponse(BaseModel):
    llm_enabled: bool
    provider_name: str
    base_url_configured: bool
    model: str


class ProductCreate(BaseModel):
    name: str = Field(min_length=2, max_length=160)
    description: str = Field(default="", max_length=2000)


class Product(BaseModel):
    id: str
    name: str
    description: str
    created_at: str


class ContextDocumentCreate(BaseModel):
    title: str = Field(min_length=2, max_length=180)
    text: str = Field(min_length=20)
    source_type: str = Field(default="manual", max_length=60)


class ContextDocument(BaseModel):
    id: str
    product_id: str
    title: str
    text: str
    source_type: str
    character_count: int
    created_at: str


class ProductKnowledgeBase(BaseModel):
    product: Product
    contexts: list[ContextDocument]
    context_count: int
    total_characters: int


class ProductReviewRequest(BaseModel):
    feature_name: str = Field(default="Untitled PRD", max_length=160)
    platform: str = Field(default="Unspecified", max_length=120)
    prd_text: str = Field(min_length=80)
    attachments: list[str] = Field(default_factory=list)
    token_budget: int = Field(default=12000, ge=1000, le=200000)
    config: ReviewConfig = Field(default_factory=ReviewConfig)


class CriterionAnalysis(BaseModel):
    criteria: str
    evaluation: EvaluationStatus
    problem_summary: str


class Finding(BaseModel):
    issue: str
    rationale: str
    section_name: str
    write_ready_replacement_text: str
    severity: str


class TokenPlan(BaseModel):
    input_tokens_estimate: int
    output_tokens_target: int
    route: TokenRoute
    model_hint: str
    budget_status: str
    compaction_actions: list[str]


class AgentStep(BaseModel):
    step: str
    tool: str
    observation: str
    hardness_score: int


class ReviewResponse(BaseModel):
    feature_name: str
    classification_level: ClassificationLevel
    classification_reason: str
    overall_assessment: ReadinessStatus
    dimensional_analysis: list[CriterionAnalysis]
    critical_blocker: str
    detailed_findings: list[Finding]
    critical_actions: list[str]
    optimizations: list[str]
    token_plan: TokenPlan
    score: int
    agent_architecture: str = "single-pass"
    agent_trace: list[AgentStep] = Field(default_factory=list)
