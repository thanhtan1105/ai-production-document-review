import os
import re
import subprocess
import sys
import argparse

# --- TOOLS ---

def get_diff():
    """Returns the code changes."""
    try:
        with open('context_file.txt', 'r') as f:
            return f.read()
    except FileNotFoundError:
        return "Error: context_file.txt not found."

def list_skills():
    """Returns a list of available review skills from the catalog."""
    catalog = []
    for root, dirs, files in os.walk('.agent-skills'):
        if 'SKILL.md' in files:
            filepath = os.path.join(root, 'SKILL.md')
            with open(filepath, 'r') as f:
                content = f.read()
                if content.startswith('---'):
                    parts = content.split('---', 2)
                    if len(parts) >= 3:
                        fm = parts[1]
                        name_match = re.search(r'^name:\s*(.+)$', fm, re.MULTILINE)
                        desc_match = re.search(r'^description:\s*(.+)$', fm, re.MULTILINE)
                        if name_match and desc_match:
                            name = name_match.group(1).strip()
                            desc = desc_match.group(1).strip()
                            catalog.append(f"Skill: {name}\nDescription: {desc}")
    if not catalog:
        return "No skills found in .agent-skills/"
    return "\n\n".join(catalog)

def read_skill(skill_name):
    """Reads the prompt of a specific review skill."""
    path = ""
    for root, dirs, files in os.walk('.agent-skills'):
        if skill_name in root and 'SKILL.md' in files:
            path = os.path.join(root, 'SKILL.md')
            break
            
    if not path:
        return f"Error: Could not find skill {skill_name}"
    
    try:
        with open(path, 'r') as f:
            prompt = f.read()
        return prompt
    except Exception as e:
        return f"Exception while reading skill: {str(e)}"

TOOLS = {
    "get_diff": get_diff,
    "list_skills": list_skills,
    "read_skill": read_skill,
}

# --- AGENT CORE ---

def get_system_prompt(target_skill):
    prompt = f"""You are an autonomous Code Review Agent acting as a specialist in '{target_skill}'.
Your goal is to analyze the provided code changes specifically focusing on '{target_skill}'.

You have access to the following tools:

1. get_diff: 
   Description: Returns the code changes that need to be reviewed.
   Usage: get_diff
   
2. list_skills:
   Description: Returns a catalog of available code review skills.
   Usage: list_skills
   
3. read_skill:
   Description: Reads the specific instructions/prompt for a skill.
   Usage: read_skill <skill_name>

4. write_report:
   Description: Writes your final review findings to the report.
   Usage: write_report <markdown_content>

5. finish:
   Description: Ends the review process. Call this ONLY after writing your report.
   Usage: finish

You MUST use the following exact format for your responses:
Thought: think about what to do next
Action: the name of the tool to use (get_diff, list_skills, read_skill, write_report, finish)
Action Input: the input to the tool (if any, omit if none. For write_report, put your markdown here)

Wait for the "Observation: ..." response before continuing.

Begin!"""
    return prompt

def call_llm(system_prompt, history):
    prompt = system_prompt + "\n\n" + history
    result = subprocess.run(
        ["ollama", "launch", "copilot", "--model", "glm-5.2:cloud", "--", prompt],
        capture_output=True, text=True
    )
    return result.stdout

def run_agent(target_skill):
    print("===========================================")
    print(f"🚀 Starting Python ReAct Agent for Skill: {target_skill}")
    print("===========================================")
    
    system_prompt = get_system_prompt(target_skill)
    history = ""
    max_steps = 10
        
    for step in range(max_steps):
        print(f"\n--- [Step {step + 1}] Thinking ---")
        response = call_llm(system_prompt, history)
        print(response)
        
        history += response + "\n"
        
        action_match = re.search(r'Action:\s*(.+)', response)
        # We use re.DOTALL to capture multi-line inputs for write_report
        action_input_match = re.search(r'Action Input:\s*(.*)', response, re.DOTALL)
        
        if action_match:
            action = action_match.group(1).strip()
            # If there's another "Thought:" or "Observation:" in the input match, we should trim it
            action_input = ""
            if action_input_match:
                raw_input = action_input_match.group(1).strip()
                # Stop parsing input if the LLM hallucinated an Observation block
                if "Observation:" in raw_input:
                    raw_input = raw_input.split("Observation:")[0].strip()
                action_input = raw_input
            
            if action == "finish":
                print("\n✅ Agent finished successfully.")
                break
                
            print(f"\n⚙️  Executing Tool: {action}")
            
            if action == "write_report":
                try:
                    with open(f'report_{target_skill}.md', 'w') as f:
                        f.write(action_input)
                    observation = f"Report saved successfully to report_{target_skill}.md"
                except Exception as e:
                    observation = f"Error saving report: {e}"
            elif action in TOOLS:
                observation = TOOLS[action](action_input) if action_input else TOOLS[action]()
            else:
                observation = f"Error: Tool '{action}' not found."
                
            # Truncate observation for console to avoid spam
            preview = str(observation)[:150].replace('\n', ' ') + "..." if len(str(observation)) > 150 else str(observation)
            print(f"👁️  Observation: {preview}")
            
            history += f"Observation: {observation}\n"
        else:
            print("\n⚠️  No action found in response. Halting to prevent loop.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ReAct Code Review Agent")
    parser.add_argument("--skill", required=True, help="The skill to execute (e.g., security, clean-code)")
    args = parser.parse_args()
    
    run_agent(args.skill)
