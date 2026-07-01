"""
LangChain ReAct Code Review Agent
Uses LangChain's ReAct agent framework with Ollama (local LLM) to perform
structured, multi-step code reviews.
"""

import os
import re
import argparse
from typing import Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate


# ---------------------------------------------------------------------------
# TOOLS
# ---------------------------------------------------------------------------

@tool
def get_diff(input: str = "") -> str:
    """
    Returns the full code diff / context that needs to be reviewed.
    Call this FIRST before doing any analysis.
    No input required.
    """
    try:
        with open("context_file.txt", "r") as f:
            target_filename = f.read().strip()
        with open(target_filename, "r") as diff_file:
            content = diff_file.read()
        if not content.strip():
            return "Warning: The diff/context file is empty. Nothing to review."
        return content
    except FileNotFoundError as e:
        return f"Error: Could not read context file. Details: {str(e)}"
    except Exception as e:
        return f"Error: Unexpected error reading diff. Details: {str(e)}"


@tool
def list_skills(input: str = "") -> str:
    """
    Returns a catalog of available code review skills loaded in .agent-skills/.
    Use this to discover what skill guidelines are available.
    No input required.
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
    Use this to load the full instructions for the skill you are executing.
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
    Call this ONCE after completing your full analysis. The report will be
    saved and included in the final combined review output.
    """
    if not markdown_content or not markdown_content.strip():
        return "Error: Cannot write an empty report. Provide your Markdown review content."

    # The skill name is stored in the environment so each review step
    # writes to its own report file.
    skill = os.environ.get("REVIEW_SKILL", "unknown")
    report_path = f"report_{skill}.md"
    try:
        with open(report_path, "w") as f:
            f.write(markdown_content.strip())
        return f"✅ Report saved successfully to '{report_path}'."
    except Exception as e:
        return f"Error saving report: {str(e)}"


# ---------------------------------------------------------------------------
# PROMPT TEMPLATE (LangChain ReAct standard format)
# ---------------------------------------------------------------------------

REACT_TEMPLATE = """\
You are an autonomous Code Review Agent specializing in '{target_skill}'.

{custom_instructions}

You have access to the following tools:

{tools}

Use the following format EXACTLY for every response:

Thought: [your reasoning about what to do next]
Action: [the tool name — must be one of: {tool_names}]
Action Input: [the input to the tool, or empty string if none]
Observation: [the result of the tool call — this will be filled in for you]
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I have completed my analysis and written the report.
Final Answer: Code review for '{target_skill}' is complete. Report has been saved.

IMPORTANT RULES:
- Always call get_diff FIRST to retrieve the code to review.
- Read the skill guidelines with read_skill before analyzing.
- Analyse the diff thoroughly against the skill's checklist.
- Write a comprehensive, structured Markdown report using write_report.
- End with Final Answer only after the report has been saved.

Begin!

{agent_scratchpad}"""


# ---------------------------------------------------------------------------
# AGENT RUNNER
# ---------------------------------------------------------------------------

def build_prompt(target_skill: str, custom_instructions: str) -> PromptTemplate:
    """Constructs the ReAct PromptTemplate with skill-specific instructions."""
    section = ""
    if custom_instructions.strip():
        section = (
            f"## Specific Review Instructions for '{target_skill}'\n"
            f"{custom_instructions.strip()}\n"
            f"---\n"
        )

    filled = REACT_TEMPLATE.replace("{custom_instructions}", section)
    return PromptTemplate.from_template(filled)


def run_agent(target_skill: str, prompt_file: str = "") -> None:
    print("===========================================")
    print(f"🚀 Starting LangChain ReAct Agent for Skill: {target_skill}")

    # --- Load custom step prompt ---
    custom_instructions = ""
    if prompt_file:
        try:
            with open(prompt_file, "r") as f:
                custom_instructions = f.read().strip()
            print(f"📝 Loaded step prompt from: {prompt_file}")
        except Exception as e:
            print(f"⚠️  Could not load step prompt from '{prompt_file}': {e}")

    print("===========================================")

    # Expose skill name to the write_report tool via environment
    os.environ["REVIEW_SKILL"] = target_skill

    # --- LLM (Ollama running locally on the self-hosted runner) ---
    ollama_model = os.environ.get("OLLAMA_MODEL", "glm-5.2:cloud")
    ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/api/chat")

    print(f"🤖 Using model: {ollama_model} @ {ollama_base_url}")

    llm = ChatOllama(
        model=ollama_model,
        base_url=ollama_base_url,
        temperature=0,          # Deterministic for code review
        num_predict=4096,       # Allow long responses for detailed reports
    )

    # --- Tools ---
    tools = [get_diff, list_skills, read_skill, write_report]

    # --- Build prompt ---
    prompt = build_prompt(target_skill, custom_instructions)

    # --- Create LangChain ReAct Agent ---
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,           # Print every Thought/Action/Observation
        max_iterations=15,      # Max ReAct loop iterations
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    # --- Run ---
    print(f"\n🔍 Starting review for skill: '{target_skill}'\n")
    try:
        result = agent_executor.invoke({
            "target_skill": target_skill,
        })
        print("\n✅ Agent completed successfully.")
        print(f"📄 Final Answer: {result.get('output', '')}")
    except Exception as e:
        print(f"\n❌ Agent encountered an error: {e}")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangChain ReAct Code Review Agent")
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
