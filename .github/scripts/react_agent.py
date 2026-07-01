import os
import re
import subprocess
import sys

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

def run_skill(skill_name):
    """Executes a specific review skill."""
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
        diff = get_diff()
        
        full_prompt = f"{prompt}\n\n=== CODE ===\n\n{diff}"
        
        print(f"      [Tool] Calling Ollama for skill: {skill_name}...")
        result = subprocess.run(
            ["ollama", "launch", "copilot", "--model", "glm-5.2:cloud", "--", full_prompt],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            return f"Error executing review: {result.stderr}"
            
        with open('final_review_report.md', 'a') as f:
            f.write(f"\n## Review by {skill_name}\n")
            f.write(result.stdout)
            f.write("\n\n")
            
        return f"Successfully executed {skill_name}. Report appended."
    except Exception as e:
        return f"Exception while running skill: {str(e)}"

TOOLS = {
    "get_diff": get_diff,
    "list_skills": list_skills,
    "run_skill": run_skill,
}

# --- AGENT CORE ---

SYSTEM_PROMPT = """You are an autonomous Code Review Orchestrator Agent.
Your goal is to analyze the provided code changes and decide which review skills to apply.
You have access to the following tools:

1. get_diff: 
   Description: Returns the code changes that need to be reviewed.
   Usage: get_diff
   
2. list_skills:
   Description: Returns a catalog of available code review skills.
   Usage: list_skills
   
3. run_skill:
   Description: Executes a specific review skill on the code and saves the report.
   Usage: run_skill <skill_name>

4. finish:
   Description: Ends the review process when you are done.
   Usage: finish

You MUST use the following exact format for your responses:
Thought: think about what to do next
Action: the name of the tool to use (one of get_diff, list_skills, run_skill, finish)
Action Input: the input to the tool (if any, omit if none)

Wait for the "Observation: ..." response before continuing.

Begin!"""

def call_llm(history):
    prompt = SYSTEM_PROMPT + "\n\n" + history
    result = subprocess.run(
        ["ollama", "launch", "copilot", "--model", "glm-5.2:cloud", "--", prompt],
        capture_output=True, text=True
    )
    return result.stdout

def run_agent():
    print("===========================================")
    print("🚀 Starting Python ReAct Agent (Tool-Capable)")
    print("===========================================")
    
    history = ""
    max_steps = 10
    
    # Initialize report
    with open('final_review_report.md', 'w') as f:
        f.write("=== AI CODE REVIEW REPORT (ReAct Agent) ===\n")
        
    for step in range(max_steps):
        print(f"\n--- [Step {step + 1}] Thinking ---")
        response = call_llm(history)
        print(response)
        
        history += response + "\n"
        
        action_match = re.search(r'Action:\s*(.+)', response)
        action_input_match = re.search(r'Action Input:\s*(.*)', response)
        
        if action_match:
            action = action_match.group(1).strip()
            action_input = action_input_match.group(1).strip() if action_input_match else ""
            
            if action == "finish":
                print("\n✅ Agent finished successfully.")
                break
                
            print(f"\n⚙️  Executing Tool: {action}('{action_input}')")
            
            if action in TOOLS:
                observation = TOOLS[action](action_input) if action_input else TOOLS[action]()
            else:
                observation = f"Error: Tool '{action}' not found."
                
            # Truncate observation for console to avoid spam
            preview = observation[:150].replace('\n', ' ') + "..." if len(observation) > 150 else observation
            print(f"👁️  Observation: {preview}")
            
            history += f"Observation: {observation}\n"
        else:
            print("\n⚠️  No action found in response. Halting to prevent loop.")
            break
            
if __name__ == "__main__":
    run_agent()
