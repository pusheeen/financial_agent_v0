import os
import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from dotenv import load_dotenv

from app.neo4j_client import get_neo4j_client

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# Allow override and fallback to reference repo data directory
FINANCIALS_DIR = Path(
    os.getenv(
        "FINANCIALS_DIR",
        str(DATA_DIR / "structured" / "financials"),
    )
)
REF_FINANCIALS_DIR = Path("/workspace/Financial_ADK_Agent_Graph_Database/data/structured/financials")


def upsert_companies_from_csv(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    records = df.to_dict("records")

    cypher = (
        "UNWIND $records AS record "
        "MERGE (c:Company {ticker: record.ticker}) "
        "SET c.name = record.company_name, c.cik = toString(record.cik)"
    )
    client = get_neo4j_client()
    client.run(cypher, {"records": records})


def upsert_financials_from_dir(financials_dir: Path) -> None:
    client = get_neo4j_client()

    for file in financials_dir.glob("*.json"):
        ticker = file.name.split("_")[0].upper()
        data = json.loads(file.read_text())
        records_to_ingest: List[Dict[str, Any]] = []
        for item in data:
            year = (item.get("date", "").split("-")[0]) if item.get("date") else None
            if not year:
                continue
            record = {
                "ticker": ticker,
                "year": year,
                "revenue": item.get("Total Revenue"),
                "netIncome": item.get("Net Income"),
                "eps": item.get("Basic EPS") or item.get("Diluted EPS"),
            }
            records_to_ingest.append(record)

        if not records_to_ingest:
            continue

        cypher = (
            "UNWIND $records AS record "
            "MATCH (c:Company {ticker: record.ticker}) "
            "MERGE (f:Financials {company: c.ticker, year: record.year}) "
            "SET f.revenue = toFloat(record.revenue), f.netIncome = toFloat(record.netIncome), f.eps = toFloat(record.eps) "
            "MERGE (c)-[:HAS_FINANCIALS]->(f)"
        )
        client.run(cypher, {"records": records_to_ingest})


if __name__ == "__main__":
    # Example usage: assumes companies.csv and JSON financials exist
    companies_csv = Path(os.getenv("COMPANIES_CSV", "/workspace/Financial_ADK_Agent_Graph_Database/companies.csv"))
    if companies_csv.exists():
        upsert_companies_from_csv(companies_csv)
    fin_dir = FINANCIALS_DIR if FINANCIALS_DIR.exists() else REF_FINANCIALS_DIR
    if fin_dir.exists():
        upsert_financials_from_dir(fin_dir)
    else:
        print(f"No financials directory found at {fin_dir}")
