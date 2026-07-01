"""
LangChain ReAct Code Review Agent
Uses LangChain's create_react_agent with ChatOllama for local LLM inference.

Stack: langchain (latest), langchain-ollama (latest), langgraph (latest)
"""

import os
import re
import argparse

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent  # noqa: LangGraph prebuilt (latest)


# ---------------------------------------------------------------------------
# TOOLS
# ---------------------------------------------------------------------------

@tool
def get_diff(input: str = "") -> str:
    """
    Returns the code diff and context to be reviewed.
    IMPORTANT: Call this with NO arguments. Do not pass any kwargs.
    Correct usage: get_diff (no input)
    Wrong usage: get_diff(format='unified') <- DO NOT do this
    """
    try:
        with open("context_file.txt", "r") as f:
            target_filename = f.read().strip()
        with open(target_filename, "r") as diff_file:
            content = diff_file.read()
        if not content.strip():
            return "Warning: The diff/context file is empty. Nothing to review."
        # Show a preview so the agent knows it received real content
        preview = content[:200].replace('\n', ' ')
        print(f"[get_diff] Read {len(content)} chars from '{target_filename}'. Preview: {preview}...")
        return content
    except FileNotFoundError as e:
        return f"Error: Could not read context file. Details: {str(e)}"
    except Exception as e:
        return f"Error: Unexpected error reading diff. Details: {str(e)}"


@tool
def list_skills(input: str = "") -> str:
    """
    Returns a catalog of available code review skills loaded in .agent-skills/.
    Use this to discover what skill guidelines are available. No input required.
    """
    catalog = []
    for root, dirs, files in os.walk(".agent-skills"):
        if "SKILL.md" in files:
            filepath = os.path.join(root, "SKILL.md")
            try:
                with open(filepath, "r") as f:
                    content = f.read()
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        fm = parts[1]
                        name_match = re.search(r"^name:\s*(.+)$", fm, re.MULTILINE)
                        desc_match = re.search(r"^description:\s*(.+)$", fm, re.MULTILINE)
                        if name_match and desc_match:
                            name = name_match.group(1).strip()
                            desc = desc_match.group(1).strip()
                            catalog.append(f"Skill: {name}\nDescription: {desc}")
            except Exception:
                pass

    if not catalog:
        return "No skills found in .agent-skills/"
    return "\n\n".join(catalog)


@tool
def read_skill(skill_name: str) -> str:
    """
    Reads the detailed guidelines for a specific review skill.
    Input: the skill name (e.g. 'security', 'clean-code', 'performance').
    """
    if not skill_name or not skill_name.strip():
        return "Error: Please provide a skill name (e.g. 'security')."

    skill_name = skill_name.strip()
    for root, dirs, files in os.walk(".agent-skills"):
        if skill_name in root and "SKILL.md" in files:
            path = os.path.join(root, "SKILL.md")
            try:
                with open(path, "r") as f:
                    return f.read()
            except Exception as e:
                return f"Exception while reading skill: {str(e)}"

    return f"Error: Could not find skill '{skill_name}'. Run list_skills to see available options."


@tool
def write_report(markdown_content: str) -> str:
    """
    Saves the final review findings to a Markdown report file.
    Input: your complete review report formatted in Markdown.
    Call this ONCE after completing your full analysis.
    """
    if not markdown_content or not markdown_content.strip():
        return "Error: Cannot write an empty report. Provide your Markdown review content."

    skill = os.environ.get("REVIEW_SKILL", "unknown")
    report_path = f"report_{skill}.md"
    try:
        with open(report_path, "w") as f:
            f.write(markdown_content.strip())
        return f"Report saved successfully to '{report_path}'."
    except Exception as e:
        return f"Error saving report: {str(e)}"


# ---------------------------------------------------------------------------
# AGENT RUNNER
# ---------------------------------------------------------------------------

def run_agent(target_skill: str, prompt_file: str = "") -> None:
    print("===========================================")
    print(f"Starting LangGraph ReAct Agent for Skill: {target_skill}")

    # --- Load custom step prompt ---
    custom_instructions = ""
    if prompt_file:
        try:
            with open(prompt_file, "r") as f:
                custom_instructions = f.read().strip()
            print(f"Loaded step prompt from: {prompt_file}")
        except Exception as e:
            print(f"Warning: Could not load step prompt from '{prompt_file}': {e}")

    print("===========================================")

    # Expose skill name to the write_report tool via environment
    os.environ["REVIEW_SKILL"] = target_skill

    # --- LLM ---
    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
    ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    print(f"Using model: {ollama_model} @ {ollama_base_url}")

    llm = ChatOllama(
        model=ollama_model,
        base_url=ollama_base_url,
        temperature=0,
        num_predict=4096,
    )

    # --- Tools ---
    tools = [get_diff, list_skills, read_skill, write_report]

    # --- LangGraph ReAct Agent ---
    agent = create_react_agent(llm, tools)

    # --- Build the user message ---
    custom_section = ""
    if custom_instructions:
        custom_section = (
            f"\n\n## Specific Review Instructions for '{target_skill}'\n"
            f"{custom_instructions}\n"
            f"---"
        )

    user_message = f"""You are a Code Review Agent specializing in '{target_skill}'.
{custom_section}

Your workflow:
1. Call get_diff to retrieve the code changes to review.
2. Call read_skill with '{target_skill}' to load the detailed review guidelines.
3. Analyze the diff thoroughly against the skill guidelines.
4. Call write_report with your complete Markdown review report.

Important: Always complete all 4 steps. Do not stop before writing the report.
Start now by calling get_diff.
"""

    print(f"\nStarting review for skill: '{target_skill}'\n")

    report_path = f"report_{target_skill}.md"

    # ---------------------------------------------------------------------------
    # RETRY LOOP — max 20 attempts
    # Each attempt runs the agent; if the report is not created (agent ran out
    # of steps or failed to call write_report), retry with a fresh invocation.
    # ---------------------------------------------------------------------------
    MAX_ATTEMPTS = 20
    success = False

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n{'='*50}")
        print(f"  Attempt {attempt}/{MAX_ATTEMPTS} — skill: '{target_skill}'")
        print(f"{'='*50}")

        # Remove stale report from a previous failed attempt so we can detect
        # whether this attempt actually wrote it.
        if os.path.exists(report_path):
            os.remove(report_path)

        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=user_message)]},
                config={"recursion_limit": 100},
            )

            # Print agent messages
            for msg in result.get("messages", []):
                msg_type = type(msg).__name__
                if hasattr(msg, "content") and msg.content:
                    content_preview = str(msg.content)[:500]
                    print(f"  [{msg_type}]: {content_preview}")

        except Exception as e:
            print(f"  [Attempt {attempt}] Agent raised an exception: {e}")
            if attempt == MAX_ATTEMPTS:
                print("All attempts exhausted with exceptions. Failing.")
                raise SystemExit(1)
            print(f"  Retrying... ({attempt + 1}/{MAX_ATTEMPTS})")
            continue

        # --- Check if the report was written this attempt ---
        if os.path.exists(report_path) and os.path.getsize(report_path) > 0:
            report_size = os.path.getsize(report_path)
            print(f"\n  Report verified: '{report_path}' ({report_size} bytes)")
            print(f"  Review for '{target_skill}' completed on attempt {attempt}/{MAX_ATTEMPTS}.")
            success = True
            break
        else:
            print(f"\n  [Attempt {attempt}] Report NOT created — agent did not call write_report.")
            if attempt < MAX_ATTEMPTS:
                print(f"  Retrying... ({attempt + 1}/{MAX_ATTEMPTS})")
            else:
                print("  All attempts exhausted. Failing.")

    if not success:
        print(f"\nERROR: '{report_path}' was not generated after {MAX_ATTEMPTS} attempts.")
        print("The agent consistently failed to complete the review. Check model/Ollama logs.")
        raise SystemExit(1)

    print(f"\nReview for '{target_skill}' completed successfully.")

    # --- Print report content to log for quick debugging ---
    separator = "=" * 60
    print(f"\n{separator}")
    print(f"  REPORT: {report_path}")
    print(f"{separator}")
    with open(report_path, "r") as f:
        print(f.read())
    print(f"{separator}\n")


# ---------------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangGraph ReAct Code Review Agent")
    parser.add_argument(
        "--skill",
        required=True,
        help="The review skill to execute (e.g. security, clean-code, performance)",
    )
    parser.add_argument(
        "--prompt-file",
        required=False,
        default="",
        help="Path to a .txt file with focused review instructions for this skill",
    )
    args = parser.parse_args()

    run_agent(args.skill, args.prompt_file)
