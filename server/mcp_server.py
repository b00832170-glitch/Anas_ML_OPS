"""
mcp_server.py - MCP Server exposing predict() as a tool for the LLM
"""

import asyncio
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.predict import predict, get_model_info

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
    ListToolsResult,
)

server = Server("credit-risk-mcp")


@server.list_tools()
async def list_tools() -> ListToolsResult:
    return ListToolsResult(
        tools=[
            Tool(
                name="predict_credit_risk",
                description=(
                    "Predicts the credit default risk of a borrower "
                    "based on personal and loan characteristics. "
                    "Returns a default probability, a risk level "
                    "(Low / Moderate / High / Very high) and a clear label."
                ),
                inputSchema={
                    "type": "object",
                    "required": [
                        "person_age", "person_income", "person_home_ownership",
                        "person_emp_length", "loan_intent", "loan_grade",
                        "loan_amnt", "loan_int_rate", "loan_percent_income",
                        "cb_person_default_on_file", "cb_person_cred_hist_length",
                    ],
                    "properties": {
                        "person_age":                  {"type": "integer",  "description": "Borrower age (18-100)"},
                        "person_income":               {"type": "number",   "description": "Annual income in dollars"},
                        "person_home_ownership":       {"type": "string",   "description": "Housing status: RENT | OWN | MORTGAGE | OTHER"},
                        "person_emp_length":           {"type": "number",   "description": "Employment length (years)"},
                        "loan_intent":                 {"type": "string",   "description": "Loan purpose: PERSONAL | EDUCATION | MEDICAL | VENTURE | HOMEIMPROVEMENT | DEBTCONSOLIDATION"},
                        "loan_grade":                  {"type": "string",   "description": "Loan grade: A | B | C | D | E | F | G"},
                        "loan_amnt":                   {"type": "number",   "description": "Loan amount"},
                        "loan_int_rate":               {"type": "number",   "description": "Interest rate (%)"},
                        "loan_percent_income":         {"type": "number",   "description": "Loan amount / annual income ratio (e.g. 0.35)"},
                        "cb_person_default_on_file":   {"type": "string",   "description": "Historical default: Y | N"},
                        "cb_person_cred_hist_length":  {"type": "integer",  "description": "Credit history length (years)"},
                    },
                },
            ),
            Tool(
                name="get_model_info",
                description=(
                    "Returns information about the prediction model: "
                    "features used, performance metrics (accuracy, ROC-AUC), "
                    "and training dataset size."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
        ]
    )


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    try:
        if name == "predict_credit_risk":
            result = predict(arguments)
            text = (
                f"Credit risk prediction result:\n"
                f"- Decision          : {result['label']}\n"
                f"- Default probability: {result['probability_default']*100:.1f}%\n"
                f"- Risk level        : {result['risk_level']}\n"
                f"- Model accuracy    : {result['model_accuracy']*100:.1f}%\n"
                f"- Model ROC-AUC     : {result['model_roc_auc']:.4f}"
            )
            return CallToolResult(content=[TextContent(type="text", text=text)])

        elif name == "get_model_info":
            info = get_model_info()
            text = json.dumps(info, indent=2, ensure_ascii=False)
            return CallToolResult(content=[TextContent(type="text", text=text)])

        else:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Unknown tool: {name}")]
            )

    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")]
        )


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())