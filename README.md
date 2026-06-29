# Automated PRD Review Framework

Local implementation of a first-pass PRD reviewer inspired by Uber's PRD review model. It uses a product-level knowledge base, classifies review depth, evaluates launch readiness across four dimensions, limits detailed findings, and reports an LLM token plan for future Copilot SDK or LiteLLM-backed execution.

## What It Does

- Creates products as review containers.
- Uploads product context documents into a product-scoped OpenKB workspace.
- Classifies PRDs into lighter, moderate, full, or full review with specialized scrutiny.
- Scores launch readiness across opportunity, scope, UX/adjacent impact, and metrics/data rigor.
- Detects blind spots such as unsupported headroom assumptions and repeated-hypothesis risk.
- Produces copy-paste replacement text for the most important gaps.
- Estimates input/output token usage and chooses a compact, standard, or advanced review route.

## Run Locally

Backend:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

## How To Evaluate A PRD

1. Start backend and frontend.
2. Open `http://127.0.0.1:5173`.
3. Paste PRD text into `PRD Content`, or use `DOCX Import` to choose a `.docx` file.
4. Adjust config:
   - `Token Budget`: review budget used for token-plan status.
   - `Max Findings`: max detailed gaps returned.
   - `Unsupported headroom`: flags growth claims without baseline evidence.
   - `Revisited hypothesis`: flags missing prior experiment/history review.
   - `Extra Sensitive Terms`: comma-separated domain terms that force specialized scrutiny.
5. Click `Run Review`.

Recommended flow:

1. Create or select a product.
2. Upload context files (`.docx`, `.md`, `.txt`, `.pdf`, `.pptx`, `.xlsx`, `.csv`, `.html`) or add manual context. Use this for business goals, architecture notes, prior experiments, metric definitions, and company PRD rules.
3. Import/paste the PRD.
4. Evaluate. The backend builds compact product context from the product OpenKB wiki before review.

## Knowledge Base Architecture

This project uses [VectifyAI/OpenKB](https://github.com/VectifyAI/OpenKB) as the knowledge-base foundation.

Each product gets its own OpenKB workspace:

```text
backend/.data/openkb/<product-id>/
  raw/
  .openkb/config.yaml
  .openkb/hashes.json
  wiki/
    index.md
    summaries/
    concepts/
    entities/
    sources/
```

On upload, the backend attempts to run:

```bash
openkb --kb-dir backend/.data/openkb/<product-id> add <file>
```

If OpenKB or LLM config is unavailable, the app writes an OpenKB-compatible fallback wiki page so local review still works. Once OpenKB is installed and configured, future uploads use the real OpenKB compiler.

## API

Create product:

```bash
curl -X POST http://127.0.0.1:8000/api/products \
  -H "Content-Type: application/json" \
  -d '{"name":"Payments Platform","description":"Checkout, billing, fraud, and marketplace settlement."}'
```

Upload knowledge-base context:

```bash
curl -F "file=@/path/to/context.docx" \
  http://127.0.0.1:8000/api/products/payments-platform/contexts/upload
```

Evaluate PRD against a product knowledge base:

```bash
curl -X POST http://127.0.0.1:8000/api/products/payments-platform/reviews \
  -H "Content-Type: application/json" \
  -d '{
    "feature_name": "New Refund Policy",
    "platform": "Web/API",
    "prd_text": "Full PRD text...",
    "token_budget": 12000,
    "config": {
      "review_mode": "heuristic",
      "max_findings": 3,
      "check_unsupported_headroom": true,
      "check_revisited_hypothesis": true,
      "extra_sensitive_terms": ["KYC", "underwriting"]
    }
  }'
```

Ad-hoc review without a product:

`POST /api/reviews`

```json
{
  "feature_name": "Smart PRD Import",
  "platform": "Web app",
  "prd_text": "Full PRD text...",
  "organizational_context": "Optional compact company context",
  "token_budget": 12000,
  "config": {
    "max_findings": 3,
    "check_unsupported_headroom": true,
    "check_revisited_hypothesis": true,
    "extra_sensitive_terms": ["KYC", "underwriting"]
  }
}
```

The response is a strict scorecard shape suitable for later replacement by a Copilot SDK/LLM reviewer.

`POST /api/reviews/extract-docx`

Multipart form upload:

```bash
curl -F "file=@/path/to/prd.docx" http://127.0.0.1:8000/api/reviews/extract-docx
```

The response returns extracted text that can be passed into `POST /api/reviews`.

## Capmial / OpenAI-Compatible LLM Config

Copy the template:

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env`:

```bash
PRD_REVIEW_LLM_PROVIDER=capmial
CAPMIAL_BASE_URL=https://YOUR_CAPMIAL_OPENAI_COMPATIBLE_BASE_URL/v1
CAPMIAL_MODEL=YOUR_MODEL_NAME
CAPMIAL_API_KEY=YOUR_API_KEY
OPENKB_MODEL=YOUR_LITELLM_MODEL_NAME
```

Then set `Review Mode` in the UI:

- `Heuristic`: local deterministic review, no LLM call.
- `Agent hardness loop`: ReAct-style tool loop. Uses Ollama native tool calling when `PRD_REVIEW_LLM_PROVIDER=ollama`, LangChain `create_agent` for OpenAI-compatible providers, then deterministic fallback if no provider is available.
- `Hybrid`: try LLM if configured, fallback to local heuristic.
- `LLM only`: require configured LLM; returns error if config is incomplete.

Check runtime config without exposing the key:

```bash
curl http://127.0.0.1:8000/api/config/runtime
```

OpenKB reads `LLM_API_KEY`; the backend maps your configured Capmial key to `LLM_API_KEY` when invoking OpenKB. If Capmial requires a custom LiteLLM provider/base URL, configure the matching OpenKB/LiteLLM settings in the generated product workspace at `backend/.data/openkb/<product-id>/.openkb/config.yaml`.

## Agent Hardness Loop

The PRD evaluator supports a ReAct-style hardness loop via `config.review_mode = "agent"`.

Loop:

1. `classify_review_hardness`: classify PRD difficulty from sensitivity, missing dimensions, findings, token budget, and PRD size.
2. `inspect_openkb_context`: inspect product OpenKB context coverage for prior experiments, metrics, and guardrails.
3. `run_dimension_reviewers`: target weak readiness dimensions.
4. `synthesize_scorecard`: produce the final strict scorecard.

With Ollama Cloud/local Ollama configured, the app calls `/api/chat` with real `tools` and records the provider-driven tool loop as `agent_architecture = "ollama-native-tool-agent"`. With an OpenAI-compatible provider configured, it attempts LangChain's maintained agent harness (`langchain.agents.create_agent`) with tools. Without provider config, it runs the same tool-shaped loop locally and returns `agent_trace` so you can audit every step.

## Future LLM Integration

The current engine is deterministic and local. To connect an LLM:

1. Keep `ReviewRequest` and `ReviewResponse` as the API contract.
2. Route `compact-classifier` to a cheap model.
3. Route `standard-reviewer` to the default review model.
4. Route `advanced-scrutiny` through a stronger model or specialized reviewer.
5. Place stable company standards in cached context, not in every prompt.
6. Use LiteLLM or a Copilot-compatible proxy as the provider gateway.

## Tests

```bash
cd backend
PYTHONPATH=. pytest
```
