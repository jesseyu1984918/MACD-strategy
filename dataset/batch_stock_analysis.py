import argparse
import asyncio
import csv
import importlib.util
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path


SKILL_SCRIPT = Path(
    r"C:\Users\jesse\.codex\skills\stock-analysis\scripts\analyze_stock.py"
)


def load_skill():
    spec = importlib.util.spec_from_file_location("stock_skill", SKILL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def analyze_ticker(skill, ticker, market_context):
    ticker = ticker.upper()
    data = skill.fetch_stock_data(ticker)
    if data is None:
        raise ValueError("invalid ticker or data unavailable")

    company_name = data.info.get("longName") or data.info.get("shortName") or ticker
    if data.asset_type == "crypto":
        crypto = skill.analyze_crypto_fundamentals(data)
        fundamentals = None
        if crypto:
            fundamentals = skill.Fundamentals(
                score=crypto.score,
                key_metrics={
                    "market_cap": crypto.market_cap,
                    "market_cap_rank": crypto.market_cap_rank,
                    "category": crypto.category,
                    "btc_correlation": crypto.btc_correlation,
                },
                explanation=crypto.explanation,
            )
        earnings = analysts = historical = earnings_timing = sector = sentiment = None
    else:
        earnings = skill.analyze_earnings_surprise(data)
        fundamentals = skill.analyze_fundamentals(data)
        analysts = skill.analyze_analyst_sentiment(data)
        historical = skill.analyze_historical_patterns(data)
        earnings_timing = skill.analyze_earnings_timing(data)
        sector = skill.analyze_sector_performance(data)
        sentiment = asyncio.run(skill.analyze_sentiment(data, skip_insider=True))

    momentum = skill.analyze_momentum(data)
    signal = skill.synthesize_signal(
        ticker=ticker,
        company_name=company_name,
        earnings=earnings,
        fundamentals=fundamentals,
        analysts=analysts,
        historical=historical,
        market_context=market_context,
        sector=sector,
        earnings_timing=earnings_timing,
        momentum=momentum,
        sentiment=sentiment,
        breaking_news=None,
        geopolitical_risk_warning=None,
        geopolitical_risk_penalty=0.0,
    )
    return asdict(signal)


def write_checkpoint(results, errors, output_path):
    payload = {
        "results": sorted(results, key=lambda row: row["ticker"]),
        "errors": sorted(errors, key=lambda row: row["ticker"]),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="combined_universe.csv")
    parser.add_argument("--output", default="combined_universe_analysis.json")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--delay", type=float, default=0.0)
    args = parser.parse_args()

    with open(args.input, newline="", encoding="utf-8-sig") as handle:
        tickers = list(dict.fromkeys(row["Symbol"].strip() for row in csv.DictReader(handle)))

    output_path = Path(args.output)
    results = []
    errors = []
    if args.retry_errors and output_path.exists():
        prior = json.loads(output_path.read_text(encoding="utf-8"))
        results = prior["results"]
        tickers = [row["ticker"] for row in prior["errors"]]

    skill = load_skill()
    market_context = skill.analyze_market_context()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(analyze_ticker, skill, ticker, market_context): ticker
            for ticker in tickers
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            ticker = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append({"ticker": ticker, "error": str(exc)})
            if args.delay:
                time.sleep(args.delay)
            if completed % 25 == 0 or completed == len(tickers):
                write_checkpoint(results, errors, output_path)
                print(
                    f"{completed}/{len(tickers)} complete; "
                    f"{len(results)} succeeded, {len(errors)} failed",
                    flush=True,
                )


if __name__ == "__main__":
    main()
