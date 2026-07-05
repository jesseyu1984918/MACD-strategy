from __future__ import annotations

from pathlib import Path
import sys

VENV_SITE_PACKAGES = Path(__file__).resolve().parent / "venv" / "Lib" / "site-packages"
if VENV_SITE_PACKAGES.exists():
    sys.path.insert(0, str(VENV_SITE_PACKAGES))

from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


POSITIVE_TERMS = {
    "beat", "beats", "surge", "surges", "growth", "strong", "stronger", "record",
    "profit", "profits", "bullish", "upgrade", "upgrades", "buyback", "expansion",
    "expand", "partnership", "launch", "raised", "raise", "outperform", "momentum",
    "rebound", "recovery", "gain", "gains", "winner", "positive",
}

NEGATIVE_TERMS = {
    "miss", "misses", "drop", "drops", "fall", "falls", "weak", "weaker", "cut",
    "cuts", "downgrade", "downgrades", "lawsuit", "probe", "investigation",
    "warning", "warns", "risk", "risks", "loss", "losses", "decline", "declines",
    "bankruptcy", "fraud", "selloff", "sell-off", "negative", "plunge", "plunges",
    "sink", "sinks", "delay", "delays", "tariff", "recall",
}


def normalize_article(article: dict) -> dict:
    content = article.get("content", article)
    return {
        "title": str(content.get("title") or ""),
        "summary": str(content.get("summary") or content.get("description") or ""),
        "published_at": content.get("pubDate") or content.get("displayTime"),
        "publisher": (content.get("provider") or {}).get("displayName", ""),
        "url": ((content.get("clickThroughUrl") or {}).get("url")) or ((content.get("canonicalUrl") or {}).get("url")) or "",
    }


def article_sentiment_value(title: str, summary: str) -> float:
    text = f"{title} {summary}".lower()
    positive_hits = sum(term in text for term in POSITIVE_TERMS)
    negative_hits = sum(term in text for term in NEGATIVE_TERMS)
    return float(positive_hits - negative_hits)


def recency_weight(published_at: str | None) -> float:
    if not published_at:
        return 1.0
    try:
        published = datetime.fromisoformat(str(published_at).replace("Z", "+00:00"))
        age_hours = max((datetime.now(timezone.utc) - published).total_seconds() / 3600, 0)
        return 1 / (1 + age_hours / 72)
    except Exception:
        return 1.0


def get_symbol_news_sentiment(symbol: str, limit: int = 8) -> dict:
    try:
        articles = yf.Ticker(symbol).news or []
    except Exception:
        articles = []

    normalized = [normalize_article(article) for article in articles[:limit]]
    if not normalized:
        return {
            "symbol": symbol,
            "sentiment_score": 5.0,
            "headline_count": 0,
            "headlines": [],
        }

    weighted_values = []
    headlines = []
    for article in normalized:
        value = article_sentiment_value(article["title"], article["summary"])
        weight = recency_weight(article["published_at"])
        weighted_values.append((value, weight))
        headlines.append(article["title"])

    total_weight = sum(weight for _, weight in weighted_values) or 1.0
    weighted_average = sum(value * weight for value, weight in weighted_values) / total_weight
    score = 5 + max(min(weighted_average * 1.5, 5), -5)
    score = round(float(score), 2)

    return {
        "symbol": symbol,
        "sentiment_score": score,
        "headline_count": len(normalized),
        "headlines": headlines[:3],
    }


def get_symbol_news_sentiment_df(symbols: list[str], limit: int = 8) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        result = get_symbol_news_sentiment(symbol, limit=limit)
        rows.append(
            {
                "Symbol": symbol,
                "NewsSentiment": result["sentiment_score"],
                "NewsHeadlineCount": result["headline_count"],
                "NewsHeadlinePreview": " | ".join(result["headlines"]),
            }
        )
    return pd.DataFrame(rows)
