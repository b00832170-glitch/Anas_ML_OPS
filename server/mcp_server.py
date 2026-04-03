"""
server/mcp_server.py - Serveur MCP qui expose predict() comme tool pour le LLM
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
    """Déclare les tools disponibles au LLM."""
    return ListToolsResult(
        tools=[
            Tool(
                name="predict_credit_risk",
                description=(
                    "Prédit le risque de défaut de crédit d'un emprunteur "
                    "à partir de ses caractéristiques personnelles et de son prêt. "
                    "Retourne une probabilité de défaut, un niveau de risque "
                    "(Faible / Modéré / Élevé / Très élevé) et un label clair."
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
                        "person_age":                  {"type": "integer",  "description": "Âge de l'emprunteur (18-100)"},
                        "person_income":               {"type": "number",   "description": "Revenu annuel en dollars"},
                        "person_home_ownership":       {"type": "string",   "description": "Statut logement : RENT | OWN | MORTGAGE | OTHER"},
                        "person_emp_length":           {"type": "number",   "description": "Ancienneté dans l'emploi (années)"},
                        "loan_intent":                 {"type": "string",   "description": "Objet du prêt : PERSONAL | EDUCATION | MEDICAL | VENTURE | HOMEIMPROVEMENT | DEBTCONSOLIDATION"},
                        "loan_grade":                  {"type": "string",   "description": "Grade du prêt : A | B | C | D | E | F | G"},
                        "loan_amnt":                   {"type": "number",   "description": "Montant du prêt"},
                        "loan_int_rate":               {"type": "number",   "description": "Taux d'intérêt en %"},
                        "loan_percent_income":         {"type": "number",   "description": "Ratio montant prêt / revenu annuel (ex: 0.35)"},
                        "cb_person_default_on_file":   {"type": "string",   "description": "Défaut historique : Y | N"},
                        "cb_person_cred_hist_length":  {"type": "integer",  "description": "Longueur historique de crédit (années)"},
                    },
                },
            ),
            Tool(
                name="get_model_info",
                description=(
                    "Retourne les informations sur le modèle de prédiction : "
                    "features utilisées, performances (accuracy, ROC-AUC), "
                    "taille du dataset d'entraînement."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
        ]
    )


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    """Exécute le tool demandé par le LLM."""
    try:
        if name == "predict_credit_risk":
            result = predict(arguments)
            text = (
                f"Résultat de la prédiction de risque crédit :\n"
                f"- Décision     : {result['label']}\n"
                f"- Probabilité de défaut : {result['probability_default']*100:.1f}%\n"
                f"- Niveau de risque      : {result['risk_level']}\n"
                f"- Modèle (Accuracy)     : {result['model_accuracy']*100:.1f}%\n"
                f"- Modèle (ROC-AUC)      : {result['model_roc_auc']:.4f}"
            )
            return CallToolResult(content=[TextContent(type="text", text=text)])

        elif name == "get_model_info":
            info = get_model_info()
            text = json.dumps(info, indent=2, ensure_ascii=False)
            return CallToolResult(content=[TextContent(type="text", text=text)])

        else:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Tool inconnu : {name}")]
            )

    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Erreur : {str(e)}")]
        )


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
