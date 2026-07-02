"""
Chunked LangChain code review runner.

The original implementation used a ReAct agent and required the model to call a
write_report tool. Some Ollama-hosted models respond with prose instead of tool
calls, which causes infinite-looking retries. This runner keeps the map-reduce
chunking strategy, but writes reports in Python after direct LLM generations.
"""

import os
import re
import argparse

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


# ---------------------------------------------------------------------------
# MAP-REDUCE CHUNKING LOGIC
# ---------------------------------------------------------------------------

def chunk_diff(payload: str, max_chars: int = 30000) -> list[str]:
    """Splits a large diff into chunks without breaking individual files."""
    if len(payload) <= max_chars:
        return [payload]

    parts = re.split(r'(?=diff --git )', payload)
    chunks = []
    current_chunk = ""

    for part in parts:
        if not part.strip():
            continue
        if len(part) > max_chars:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            chunks.extend(split_large_part(part, max_chars=max_chars))
        elif len(current_chunk) + len(part) > max_chars and current_chunk:
            chunks.append(current_chunk)
            current_chunk = part
        else:
            current_chunk += part

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def split_large_part(part: str, max_chars: int) -> list[str]:
    """Splits a single oversized file diff on line boundaries."""
    chunks = []
    current_lines = []
    current_len = 0

    for line in part.splitlines(keepends=True):
        if current_lines and current_len + len(line) > max_chars:
            chunks.append("".join(current_lines))
            current_lines = []
            current_len = 0
        current_lines.append(line)
        current_len += len(line)

    if current_lines:
        chunks.append("".join(current_lines))

    return chunks


def read_text_file(path: str, description: str) -> str:
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {description} from '{path}': {e}")
        raise SystemExit(1)


def read_optional_text_file(path: str, description: str) -> str:
    if not path:
        return ""
    try:
        with open(path, "r") as f:
            content = f.read().strip()
        print(f"Loaded {description} from: {path}")
        return content
    except Exception as e:
        print(f"Warning: Could not load {description} from '{path}': {e}")
        return ""


def read_skill_content(skill_name: str) -> str:
    for root, dirs, files in os.walk(".agent-skills"):
        if "SKILL.md" not in files:
            continue
        if os.path.basename(root) == skill_name or skill_name in root.split(os.sep):
            path = os.path.join(root, "SKILL.md")
            content = read_text_file(path, f"skill '{skill_name}'")
            print(f"Loaded skill instructions from: {path}")
            return content.strip()

    print(f"Warning: Could not find skill '{skill_name}' in .agent-skills/.")
    return ""


def message_to_text(message) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content).strip()


def build_review_prompt(
    target_skill: str,
    review_target: str,
    custom_instructions: str,
    skill_content: str,
    chunk: str,
    chunk_number: int,
    total_chunks: int,
) -> list:
    target_context = ""
    if review_target in ["backend", "frontend"]:
        target_context = (
            f"You are reviewing the {review_target.upper()} codebase only. "
            f"Focus on {review_target}-specific technologies and ignore unrelated files."
        )

    return [
        SystemMessage(
            content=(
                "You are a senior code review specialist. Return only a Markdown "
                "review report. Do not describe your process. Do not call tools."
            )
        ),
        HumanMessage(
            content=f"""Review skill: {target_skill}
{target_context}

Specific review instructions:
{custom_instructions or "(No extra prompt provided.)"}

Skill guidelines:
{skill_content or "(No skill file was found; use standard code review practice for this skill.)"}

Chunk context:
- Chunk {chunk_number} of {total_chunks}
- Report only issues visible in this chunk.
- Quote only code that appears in this chunk.
- If no issues are found, return exactly: Pass

Diff/context chunk:
```diff
{chunk}
```
"""
        ),
    ]


def build_reduce_prompt(target_skill: str, reports: list[str]) -> list:
    return [
        SystemMessage(
            content=(
                "You consolidate chunked code review reports. Return only Markdown. "
                "Preserve concrete findings, remove duplicates, and do not invent issues."
            )
        ),
        HumanMessage(
            content=f"""Merge these chunk-level reports for review skill '{target_skill}'.

Requirements:
- Preserve file paths, line numbers, severities, code snippets, and fixes where present.
- Remove duplicate findings across chunks.
- If every chunk passed, return a concise pass report.
- End with a summary table and overall score when the chunk reports contain enough detail.

Chunk reports:
{chr(10).join(reports)}
"""
        ),
    ]


def invoke_llm_with_retries(llm: ChatOllama, messages: list, label: str, max_attempts: int) -> str:
    for attempt in range(1, max_attempts + 1):
        try:
            response = llm.invoke(messages)
            markdown = message_to_text(response)
            preview = markdown[:500].replace("\n", " ")
            print(f"  [{label} attempt {attempt}] Response preview: {preview}")
            if markdown:
                return markdown
            print(f"  [{label} attempt {attempt}] Empty response.")
        except Exception as e:
            print(f"  [{label} attempt {attempt}] LLM raised an exception: {e}")

        if attempt < max_attempts:
            print(f"  Retrying {label}... ({attempt + 1}/{max_attempts})")

    raise RuntimeError(f"Failed to generate non-empty output for {label} after {max_attempts} attempts.")


# ---------------------------------------------------------------------------
# AGENT RUNNER
# ---------------------------------------------------------------------------

def run_agent(target_skill: str, prompt_file: str = "") -> None:
    print("===========================================")
    print(f"Starting chunked LangChain reviewer for Skill: {target_skill}")

    # --- Load custom step prompt ---
    custom_instructions = read_optional_text_file(prompt_file, "step prompt")
    skill_content = read_skill_content(target_skill)

    print("===========================================")

    # --- Read Full Payload from Context File ---
    target_filename = read_text_file("context_file.txt", "context file pointer").strip()
    full_payload = read_text_file(target_filename, "review context")

    max_chunk_chars = int(os.environ.get("REVIEW_MAX_CHUNK_CHARS", "30000"))
    chunks = chunk_diff(full_payload, max_chars=max_chunk_chars)
    print(f"Diff chunked into {len(chunks)} parts.")

    # --- LLM Setup ---
    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
    ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    print(f"Using model: {ollama_model} @ {ollama_base_url}")

    llm = ChatOllama(
        model=ollama_model,
        base_url=ollama_base_url,
        temperature=0,
        num_predict=4096,
        num_ctx=128000,
    )

    review_target = os.environ.get("REVIEW_TARGET", "all")

    print(f"\nStarting chunked review for skill: '{target_skill}'\n")

    report_path = f"report_{target_skill}.md"
    if os.path.exists(report_path):
        os.remove(report_path)

    final_reports = []
    max_attempts = int(os.environ.get("REVIEW_MAX_ATTEMPTS", "3"))

    for i, chunk in enumerate(chunks):
        print(f"\n{'='*50}")
        print(f"  Processing Chunk {i+1}/{len(chunks)} - skill: '{target_skill}'")
        print(f"{'='*50}")

        preview = chunk[:200].replace("\n", " ")
        print(f"  Chunk size: {len(chunk)} chars. Preview: {preview}...")

        messages = build_review_prompt(
            target_skill=target_skill,
            review_target=review_target,
            custom_instructions=custom_instructions,
            skill_content=skill_content,
            chunk=chunk,
            chunk_number=i + 1,
            total_chunks=len(chunks),
        )

        try:
            report = invoke_llm_with_retries(
                llm,
                messages,
                label=f"chunk {i+1}",
                max_attempts=max_attempts,
            )
        except RuntimeError as e:
            print(f"\nERROR: {e}")
            raise SystemExit(1)

        final_reports.append(f"### Review Part {i+1}\n\n{report}")
        print(f"\n  Report for chunk {i+1} captured successfully.")

    print(f"\nReview for '{target_skill}' completed successfully across {len(chunks)} chunks.")

    # --- Write Aggregated Report ---
    aggregated_markdown = f"# Code Review Report: {target_skill}\n\n" + "\n\n---\n\n".join(final_reports)
    if len(final_reports) > 1 and os.environ.get("REVIEW_SKIP_REDUCE", "false").lower() != "true":
        print("\nReducing chunk reports into one consolidated report...")
        try:
            reduced_report = invoke_llm_with_retries(
                llm,
                build_reduce_prompt(target_skill, final_reports),
                label="reduce",
                max_attempts=max_attempts,
            )
            aggregated_markdown = f"# Code Review Report: {target_skill}\n\n{reduced_report}"
        except RuntimeError as e:
            print(f"Warning: {e}. Falling back to concatenated chunk reports.")

    try:
        with open(report_path, "w") as f:
            f.write(aggregated_markdown)
    except Exception as e:
        print(f"Error saving final aggregated report: {e}")
        raise SystemExit(1)

    # --- Print report content to log for quick debugging ---
    separator = "=" * 60
    print(f"\n{separator}")
    print(f"  AGGREGATED REPORT: {report_path}")
    print(f"{separator}")
    print(aggregated_markdown)
    print(f"{separator}\n")


# ---------------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunked LangChain code review runner")
    parser.add_argument(
        "--skill",
        type=str,
        required=True,
        help="The specific review skill to apply (e.g. coding-standards, security).",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default="",
        help="Path to the custom step prompt text file.",
    )
    args = parser.parse_args()

    run_agent(args.skill, args.prompt_file)
