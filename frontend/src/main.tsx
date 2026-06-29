import { StrictMode, useEffect, useState, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import { AlertTriangle, Database, FileText, Gauge, Play, Plus, ShieldCheck, Target, Upload, WalletCards } from "lucide-react";
import "./styles.css";

type Product = {
  id: string;
  name: string;
  description: string;
  created_at: string;
};

type ContextDocument = {
  id: string;
  product_id: string;
  title: string;
  text: string;
  source_type: string;
  character_count: number;
  created_at: string;
};

type Criterion = {
  criteria: string;
  evaluation: "Looks Good" | "Needs Review";
  problem_summary: string;
};

type Finding = {
  issue: string;
  rationale: string;
  section_name: string;
  write_ready_replacement_text: string;
  severity: string;
};

type ReviewResponse = {
  feature_name: string;
  classification_level: string;
  classification_reason: string;
  overall_assessment: string;
  dimensional_analysis: Criterion[];
  critical_blocker: string;
  detailed_findings: Finding[];
  critical_actions: string[];
  optimizations: string[];
  token_plan: {
    input_tokens_estimate: number;
    output_tokens_target: number;
    route: string;
    model_hint: string;
    budget_status: string;
    compaction_actions: string[];
  };
  score: number;
  agent_architecture: string;
  agent_trace: {
    step: string;
    tool: string;
    observation: string;
    hardness_score: number;
  }[];
};

const samplePrd = `Problem:
Product teams currently paste requirements from scattered docs and lose review history.

Hypothesis:
If we provide a structured PRD review workflow, PMs will identify missing scope and metrics before engineering planning.

Scope:
- Paste or upload PRD text.
- Classify review depth.
- Generate launch-readiness scorecard.
- Provide write-ready replacement text.

User Experience:
The reviewer must show the most severe blocker first, then only the top three findings.
Edge cases include empty PRDs, very long documents, and missing metrics.

Metrics:
Success metric: percentage of reviewed PRDs with all four readiness dimensions complete.
Guardrail metric: average review output must stay under 1,200 tokens.`;

function App() {
  const [products, setProducts] = useState<Product[]>([]);
  const [selectedProductId, setSelectedProductId] = useState("");
  const [contexts, setContexts] = useState<ContextDocument[]>([]);
  const [productName, setProductName] = useState("AI PRD Reviewer");
  const [productDescription, setProductDescription] = useState("Automated first-pass PRD review system with product knowledge base and token-aware scorecards.");
  const [manualContextTitle, setManualContextTitle] = useState("Company PRD Standards");
  const [manualContextText, setManualContextText] = useState("Every launch needs explicit scope, non-goals, success metrics, guardrails, adjacent workflow owner approval, and prior experiment review.");
  const [featureName, setFeatureName] = useState("Smart PRD Import");
  const [platform, setPlatform] = useState("Web app");
  const [tokenBudget, setTokenBudget] = useState(12000);
  const [reviewMode, setReviewMode] = useState<"heuristic" | "agent" | "hybrid" | "llm">("heuristic");
  const [maxFindings, setMaxFindings] = useState(3);
  const [checkHeadroom, setCheckHeadroom] = useState(true);
  const [checkRevisited, setCheckRevisited] = useState(true);
  const [extraSensitiveTerms, setExtraSensitiveTerms] = useState("");
  const [prdText, setPrdText] = useState(samplePrd);
  const [review, setReview] = useState<ReviewResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const selectedProduct = products.find((product) => product.id === selectedProductId);

  useEffect(() => {
    void loadProducts();
  }, []);

  useEffect(() => {
    if (selectedProductId) {
      void loadContexts(selectedProductId);
    } else {
      setContexts([]);
    }
  }, [selectedProductId]);

  async function loadProducts() {
    const result = await fetch("/api/products");
    const payload: Product[] = await result.json();
    setProducts(payload);
    if (!selectedProductId && payload[0]) {
      setSelectedProductId(payload[0].id);
    }
  }

  async function loadContexts(productId: string) {
    const result = await fetch(`/api/products/${productId}/contexts`);
    if (result.ok) {
      setContexts(await result.json());
    }
  }

  async function createProduct() {
    setLoading(true);
    setError("");
    try {
      const result = await fetch("/api/products", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: productName, description: productDescription })
      });
      if (!result.ok) {
        throw new Error(`Create product failed with HTTP ${result.status}`);
      }
      const product: Product = await result.json();
      await loadProducts();
      setSelectedProductId(product.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown product error");
    } finally {
      setLoading(false);
    }
  }

  async function addManualContext() {
    if (!selectedProductId) {
      setError("Create or select a product first.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const result = await fetch(`/api/products/${selectedProductId}/contexts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: manualContextTitle, text: manualContextText, source_type: "manual" })
      });
      if (!result.ok) {
        throw new Error(`Add context failed with HTTP ${result.status}`);
      }
      await loadContexts(selectedProductId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown context error");
    } finally {
      setLoading(false);
    }
  }

  async function uploadContext(file: File | null) {
    if (!file || !selectedProductId) {
      return;
    }
    setLoading(true);
    setError("");
    try {
      const form = new FormData();
      form.append("file", file);
      const result = await fetch(`/api/products/${selectedProductId}/contexts/upload`, {
        method: "POST",
        body: form
      });
      if (!result.ok) {
        const payload = await result.json().catch(() => null);
        throw new Error(payload?.detail ?? `Context upload failed with HTTP ${result.status}`);
      }
      await loadContexts(selectedProductId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown upload error");
    } finally {
      setLoading(false);
    }
  }

  async function importPrdDocx(file: File | null) {
    if (!file) {
      return;
    }
    setLoading(true);
    setError("");
    try {
      const form = new FormData();
      form.append("file", file);
      const result = await fetch("/api/reviews/extract-docx", {
        method: "POST",
        body: form
      });
      if (!result.ok) {
        const payload = await result.json().catch(() => null);
        throw new Error(payload?.detail ?? `PRD import failed with HTTP ${result.status}`);
      }
      const payload: { filename: string; text: string } = await result.json();
      const parsed = extractPrdMetadata(payload.text);
      setFeatureName(parsed.featureName || payload.filename.replace(/\.docx$/i, ""));
      if (parsed.platform) {
        setPlatform(parsed.platform);
      }
      setPrdText(parsed.body);
      setReview(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown DOCX import error");
    } finally {
      setLoading(false);
    }
  }

  async function submitReview() {
    if (!selectedProductId) {
      setError("Create or select a product before evaluating a PRD.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const result = await fetch(`/api/products/${selectedProductId}/reviews`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          feature_name: featureName,
          platform,
          prd_text: extractPrdMetadata(prdText).body,
          token_budget: tokenBudget,
          config: {
            review_mode: reviewMode,
            max_findings: maxFindings,
            check_unsupported_headroom: checkHeadroom,
            check_revisited_hypothesis: checkRevisited,
            extra_sensitive_terms: extraSensitiveTerms
              .split(",")
              .map((term) => term.trim())
              .filter(Boolean)
          }
        })
      });
      if (!result.ok) {
        const payload = await result.json().catch(() => null);
        throw new Error(payload?.detail ?? `Review failed with HTTP ${result.status}`);
      }
      setReview(await result.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown review error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="workflow">
      <section className="builder-pane">
        <div className="topbar">
          <div>
            <p className="eyebrow">Automated PRD Review Framework</p>
            <h1>Product Knowledge Review</h1>
          </div>
          <button className="primary-button" onClick={submitReview} disabled={loading || !selectedProductId}>
            <Play size={18} />
            {loading ? "Working" : "Evaluate"}
          </button>
        </div>

        <div className="step-band">
          <StepBadge index="1" title="Create product" active />
          <StepBadge index="2" title="Upload context" active={Boolean(selectedProductId)} />
          <StepBadge index="3" title="Evaluate PRD" active={contexts.length > 0} />
        </div>

        <section className="panel">
          <h2>1. Product</h2>
          <div className="controls-grid">
            <label>
              Product Name
              <input value={productName} onChange={(event) => setProductName(event.target.value)} />
            </label>
            <label>
              Existing Product
              <select value={selectedProductId} onChange={(event) => setSelectedProductId(event.target.value)}>
                <option value="">Select product</option>
                {products.map((product) => (
                  <option value={product.id} key={product.id}>{product.name}</option>
                ))}
              </select>
            </label>
            <button className="secondary-button" onClick={createProduct} disabled={loading}>
              <Plus size={17} />
              Create
            </button>
          </div>
          <label>
            Product Description
            <textarea className="short-textarea" value={productDescription} onChange={(event) => setProductDescription(event.target.value)} />
          </label>
          {selectedProduct ? <p className="selected-product">Selected: <strong>{selectedProduct.name}</strong></p> : null}
        </section>

        <section className="panel">
          <h2>2. Knowledge Base</h2>
          <div className="context-actions">
            <label className="file-import">
              Upload Context
              <span>
                <Upload size={18} />
                .docx / .md / .txt
                <input
                  type="file"
                  accept=".docx,.md,.txt,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain,text/markdown"
                  onChange={(event) => void uploadContext(event.target.files?.[0] ?? null)}
                />
              </span>
            </label>
            <button className="secondary-button" onClick={addManualContext} disabled={loading || !selectedProductId}>
              <Database size={17} />
              Add Manual Context
            </button>
          </div>
          <label>
            Context Title
            <input value={manualContextTitle} onChange={(event) => setManualContextTitle(event.target.value)} />
          </label>
          <label>
            Context Text
            <textarea className="short-textarea" value={manualContextText} onChange={(event) => setManualContextText(event.target.value)} />
          </label>
          <div className="context-list">
            {contexts.map((context) => (
              <div className="context-item" key={context.id}>
                <strong>{context.title}</strong>
                <span>{context.source_type} · {context.character_count.toLocaleString()} chars</span>
              </div>
            ))}
            {!contexts.length ? <p className="muted">No context uploaded yet.</p> : null}
          </div>
        </section>

        <section className="panel">
          <h2>3. PRD Evaluation</h2>
          <div className="controls-grid">
            <label>
              Feature
              <input value={featureName} onChange={(event) => setFeatureName(event.target.value)} />
            </label>
            <label>
              Platform
              <input value={platform} onChange={(event) => setPlatform(event.target.value)} />
            </label>
            <label>
              Review Mode
              <select value={reviewMode} onChange={(event) => setReviewMode(event.target.value as "heuristic" | "agent" | "hybrid" | "llm")}>
                <option value="heuristic">Heuristic</option>
                <option value="agent">Agent hardness loop</option>
                <option value="hybrid">Hybrid</option>
                <option value="llm">LLM only</option>
              </select>
            </label>
          </div>
          <div className="config-panel">
            <label className="file-import">
              Import PRD
              <span>
                <FileText size={18} />
                Choose .docx
                <input
                  type="file"
                  accept=".docx,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                  onChange={(event) => void importPrdDocx(event.target.files?.[0] ?? null)}
                />
              </span>
            </label>
            <label>
              Token Budget
              <input type="number" min={1000} max={200000} value={tokenBudget} onChange={(event) => setTokenBudget(Number(event.target.value))} />
            </label>
            <label>
              Max Findings
              <input type="number" min={1} max={10} value={maxFindings} onChange={(event) => setMaxFindings(Number(event.target.value))} />
            </label>
          </div>
          <div className="config-panel two-checks">
            <label className="checkbox-label">
              <input type="checkbox" checked={checkHeadroom} onChange={(event) => setCheckHeadroom(event.target.checked)} />
              Unsupported headroom
            </label>
            <label className="checkbox-label">
              <input type="checkbox" checked={checkRevisited} onChange={(event) => setCheckRevisited(event.target.checked)} />
              Revisited hypothesis
            </label>
          </div>
          <label>
            Extra Sensitive Terms
            <input placeholder="e.g. credit score, KYC, underwriting" value={extraSensitiveTerms} onChange={(event) => setExtraSensitiveTerms(event.target.value)} />
          </label>
          <label className="prd-box">
            PRD Content
            <textarea value={prdText} onChange={(event) => setPrdText(event.target.value)} />
          </label>
        </section>

        {error ? <div className="error">{error}</div> : null}
      </section>

      <section className="review-pane">
        {review ? <Scorecard review={review} /> : <EmptyState />}
      </section>
    </main>
  );
}

function extractPrdMetadata(text: string): { featureName: string; platform: string; body: string } {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  let featureName = "";
  let platform = "";
  const bodyLines: string[] = [];

  for (const line of lines) {
    const featureMatch = line.match(/^\s*feature\s*name\s*:\s*(.+)\s*$/i);
    const platformMatch = line.match(/^\s*platform\s*:\s*(.+)\s*$/i);
    if (!featureName && featureMatch) {
      featureName = featureMatch[1].trim();
      continue;
    }
    if (!platform && platformMatch) {
      platform = platformMatch[1].trim();
      continue;
    }
    bodyLines.push(line);
  }

  return {
    featureName,
    platform,
    body: bodyLines.join("\n").replace(/^\s+/, "").trimEnd(),
  };
}

function StepBadge({ index, title, active }: { index: string; title: string; active: boolean }) {
  return (
    <div className={active ? "step active" : "step"}>
      <span>{index}</span>
      <strong>{title}</strong>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="empty-state">
      <ShieldCheck size={42} />
      <h2>Ready to evaluate</h2>
      <p>Create/select a product, add knowledge-base context, then evaluate a PRD against that product context.</p>
    </div>
  );
}

function Scorecard({ review }: { review: ReviewResponse }) {
  const statusClass = review.overall_assessment === "Ready" ? "ready" : review.overall_assessment === "Not Ready" ? "blocked" : "caveat";
  return (
    <div className="scorecard">
      <header className="scorecard-header">
        <div>
          <p className="eyebrow">PRD Review Scorecard</p>
          <h2>{review.feature_name}</h2>
          <p className="classification">{review.classification_level} - {review.classification_reason}</p>
        </div>
        <div className={`status-pill ${statusClass}`}>{review.overall_assessment}</div>
      </header>

      <div className="metrics-row">
        <Metric icon={<Gauge size={20} />} label="Score" value={`${review.score}/100`} />
        <Metric icon={<WalletCards size={20} />} label="Route" value={review.token_plan.route} />
        <Metric icon={<Target size={20} />} label="Token Estimate" value={`${review.token_plan.input_tokens_estimate + review.token_plan.output_tokens_target}`} />
      </div>

      <section className="blocker">
        <AlertTriangle size={20} />
        <div>
          <h3>Critical Blocker</h3>
          <p>{review.critical_blocker}</p>
        </div>
      </section>

      <section>
        <h3>Dimensional Analysis</h3>
        <div className="analysis-table">
          {review.dimensional_analysis.map((item) => (
            <div className="analysis-row" key={item.criteria}>
              <strong>{item.criteria}</strong>
              <span className={item.evaluation === "Looks Good" ? "good" : "needs"}>{item.evaluation}</span>
              <p>{item.problem_summary}</p>
            </div>
          ))}
        </div>
      </section>

      <section>
        <h3>Detailed Findings & Fixes</h3>
        <div className="findings">
          {review.detailed_findings.map((finding) => (
            <article className="finding" key={finding.issue}>
              <div className="finding-title">
                <strong>{finding.issue}</strong>
                <span>{finding.severity}</span>
              </div>
              <p>{finding.rationale}</p>
              <div className="replacement">
                <span>{finding.section_name}</span>
                <p>{finding.write_ready_replacement_text}</p>
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="action-grid">
        <ActionList title="Critical Blockers" items={review.critical_actions} />
        <ActionList title="Optimizations" items={review.optimizations} />
      </section>

      <section className="token-plan">
        <h3>LLM Token Plan</h3>
        <p><strong>{review.token_plan.budget_status}</strong> - {review.token_plan.model_hint}</p>
        <ul>
          {review.token_plan.compaction_actions.map((action) => <li key={action}>{action}</li>)}
        </ul>
      </section>

      <section className="agent-trace">
        <h3>Agent Architecture</h3>
        <p><strong>{review.agent_architecture}</strong></p>
        {review.agent_trace.length ? (
          <div className="trace-list">
            {review.agent_trace.map((step) => (
              <div className="trace-item" key={`${step.step}-${step.tool}`}>
                <span>{step.step}</span>
                <strong>{step.tool}</strong>
                <em>Hardness {step.hardness_score}</em>
                <p>{step.observation}</p>
              </div>
            ))}
          </div>
        ) : (
          <p>No agent loop trace for this mode.</p>
        )}
      </section>
    </div>
  );
}

function Metric({ icon, label, value }: { icon: ReactNode; label: string; value: string }) {
  return (
    <div className="metric">
      {icon}
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    </div>
  );
}

function ActionList({ title, items }: { title: string; items: string[] }) {
  return (
    <div>
      <h3>{title}</h3>
      <ul>
        {items.map((item) => <li key={item}>{item}</li>)}
      </ul>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
