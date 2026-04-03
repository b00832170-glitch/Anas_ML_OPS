"""
chat.py - Credit Risk AI Assistant powered by Groq LLM + ML Model
The LLM automatically calls the ML model as a tool to answer questions.
Run: python chat.py
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ml.predict import predict, get_model_info

load_dotenv()

# ── Colors ─────────────────────────────────────────────────────────────────────
BLUE   = "\033[94m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

BANNER = f"""
{BOLD}{BLUE}╔══════════════════════════════════════════════════════════╗
║         CREDIT RISK AI ASSISTANT                         ║
║         Groq LLM  +  RandomForest ML Model               ║
╚══════════════════════════════════════════════════════════╝{RESET}

{YELLOW}Ask me anything about credit risk!
I will automatically call the ML model when needed.{RESET}

Examples:
  "Is a 25-year-old renter with $50k income and a $10k personal loan risky?"
  "Analyze this borrower: 35 years old, $80k income, mortgage, grade A loan"
  "What is the model accuracy?"

Type {BOLD}'quit'{RESET} to exit.
"""

# ── Tool definitions (MCP-style) ───────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "predict_credit_risk",
            "description": (
                "Predicts the credit default risk of a borrower based on personal "
                "and loan characteristics. Returns prediction, default probability, "
                "and risk level (Low / Moderate / High / Very high)."
            ),
            "parameters": {
                "type": "object",
                "required": [
                    "person_age", "person_income", "person_home_ownership",
                    "person_emp_length", "loan_intent", "loan_grade",
                    "loan_amnt", "loan_int_rate", "loan_percent_income",
                    "cb_person_default_on_file", "cb_person_cred_hist_length",
                ],
                "properties": {
                    "person_age":                  {"type": "integer", "description": "Borrower age (18-100)"},
                    "person_income":               {"type": "number",  "description": "Annual income in USD"},
                    "person_home_ownership":       {"type": "string",  "description": "RENT | OWN | MORTGAGE | OTHER"},
                    "person_emp_length":           {"type": "number",  "description": "Employment length in years"},
                    "loan_intent":                 {"type": "string",  "description": "PERSONAL | EDUCATION | MEDICAL | VENTURE | HOMEIMPROVEMENT | DEBTCONSOLIDATION"},
                    "loan_grade":                  {"type": "string",  "description": "A | B | C | D | E | F | G"},
                    "loan_amnt":                   {"type": "number",  "description": "Loan amount in USD"},
                    "loan_int_rate":               {"type": "number",  "description": "Interest rate (%)"},
                    "loan_percent_income":         {"type": "number",  "description": "Loan amount / annual income ratio"},
                    "cb_person_default_on_file":   {"type": "string",  "description": "Historical default: Y | N"},
                    "cb_person_cred_hist_length":  {"type": "integer", "description": "Credit history length in years"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_info",
            "description": "Returns performance metrics and information about the ML model.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a credit risk AI assistant. You have access to a 
RandomForest machine learning model trained on 32,581 loan records.

When a user asks about a borrower's credit risk or loan default probability, 
you MUST call the predict_credit_risk tool with the borrower's information.
If any information is missing, ask the user for it before calling the tool.

When a user asks about the model's performance or accuracy, call get_model_info.

After getting the tool result, explain it clearly in plain English:
- State whether the borrower is likely to default or not
- Mention the default probability and risk level
- Give a brief recommendation (approve / review / decline)
- Be concise and professional
"""

# ── Tool execution ─────────────────────────────────────────────────────────────
def execute_tool(name: str, arguments: dict) -> str:
    try:
        if name == "predict_credit_risk":
            result = predict(arguments)
            return json.dumps(result)
        elif name == "get_model_info":
            info = get_model_info()
            return json.dumps(info)
        else:
            return json.dumps({"error": f"Unknown tool: {name}"})
    except Exception as e:
        return json.dumps({"error": str(e)})

# ── Main chat loop ─────────────────────────────────────────────────────────────
def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print(f"{RED}Error: GROQ_API_KEY not found in .env file.{RESET}")
        print("Create a .env file with: GROQ_API_KEY=your_key_here")
        sys.exit(1)

    client = Groq(api_key=api_key)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(BANNER)

    while True:
        # Get user input
        try:
            user_input = input(f"{BOLD}{BLUE}You: {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{YELLOW}Goodbye!{RESET}\n")
            break

        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "bye"]:
            print(f"\n{YELLOW}Goodbye!{RESET}\n")
            break

        messages.append({"role": "user", "content": user_input})

        # ── Agentic loop: LLM calls tools until it has a final answer ──────────
        while True:
            print(f"{YELLOW}  Thinking...{RESET}", end="\r")

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=1024,
            )

            message = response.choices[0].message

            # No tool call → final answer
            if not message.tool_calls:
                print(" " * 20, end="\r")  # clear "Thinking..."
                print(f"\n{BOLD}{GREEN}Assistant:{RESET} {message.content}\n")
                messages.append({"role": "assistant", "content": message.content})
                break

            # Tool call → execute and feed result back
            messages.append(message)

            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                print(f"{YELLOW}  → Calling ML tool: {name}...{RESET}", end="\r")
                result = execute_tool(name, args)
                print(" " * 50, end="\r")  # clear line

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_call.id,
                    "content":      result,
                })


if __name__ == "__main__":
    main()
