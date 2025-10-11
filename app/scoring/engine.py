"""
Scoring engine for the deepAlpha investment copilot.

The engine reads locally cached fundamentals (from the ingestion pipeline) and
enriches them with live Yahoo Finance data to compute category scores along the
dimensions defined in the product PRD.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from statistics import mean, pvariance
from typing import Dict, List, Optional, Tuple

import json
import numpy as np
import pandas as pd
import re
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


__all__ = ["compute_company_scores", "ScoreComputationError"]


class ScoreComputationError(Exception):
    """Raised when score computation fails for a ticker."""


DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
FINANCIALS_DIR = DATA_ROOT / "structured" / "financials"
EARNINGS_DIR = DATA_ROOT / "structured" / "earnings"
PRICES_DIR = DATA_ROOT / "structured" / "prices"
REPORTS_DIR = DATA_ROOT / "reports"
REDDIT_DIR = DATA_ROOT / "unstructured" / "reddit"

CRITICAL_PATH_MAP: Dict[str, float] = {
    "NVDA": 10.0,
    "TSM": 9.0,
    "SMCI": 8.5,
    "AVGO": 8.5,
    "MU": 7.5,
    "VRT": 7.0,
    "CCJ": 8.0,
    "NXE": 7.5,
    "OKLO": 7.0,
    "SMR": 7.0,
    "VST": 6.5,
    "QS": 6.0,
    "EOSE": 6.0,
    "CIFR": 6.0,
    "RIOT": 6.0,
    "IREN": 6.0,
    "RR": 5.5,
    "INOD": 5.0,
}

SECTOR_GROWTH_BONUS = {
    "Semiconductors": 10.0,
    "Semiconductor": 10.0,
    "Semiconductor Equipment & Materials": 9.0,
    "Technology": 8.5,
    "Information Technology Services": 7.5,
    "Utilities": 6.0,
    "Financial Services": 6.5,
    "Basic Materials": 6.0,
    "Industrials": 6.5,
    "Energy": 7.0,
}

sentiment_analyzer = SentimentIntensityAnalyzer()


def extract_year_from_summary(summary: Optional[str]) -> Optional[int]:
    if not summary:
        return None
    match = re.search(r"(19|20)\d{2}", summary)
    if match:
        return int(match.group())
    return None


@lru_cache(maxsize=1)
def _load_latest_ceo_summary_df() -> pd.DataFrame:
    files = sorted(REPORTS_DIR.glob("ceo_summary_*.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.read_csv(files[-1])
    df["ticker"] = df["ticker"].str.upper()
    return df


def load_ceo_profile(ticker: str) -> Optional[Dict[str, object]]:
    df = _load_latest_ceo_summary_df()
    if df.empty:
        return None
    row = df[df["ticker"] == ticker.upper()]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


@lru_cache(maxsize=1)
def load_latest_reddit_summary() -> Dict[str, dict]:
    if not REDDIT_DIR.exists():
        return {}
    candidates = list(REDDIT_DIR.glob("reddit_summary*.json"))
    if not candidates:
        return {}
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        with latest.open() as f:
            data = json.load(f)
            if isinstance(data, dict):
                return {k.upper(): v for k, v in data.items()}
    except Exception:
        return {}
    return {}


def format_currency(value: Optional[float]) -> Optional[str]:
    if value is None or value != value:
        return None
    abs_value = abs(value)
    if abs_value >= 1e12:
        return f"${value/1e12:.1f}T"
    if abs_value >= 1e9:
        return f"${value/1e9:.1f}B"
    if abs_value >= 1e6:
        return f"${value/1e6:.1f}M"
    return f"${value:,.0f}"


def summarize_company(info: dict, fin_df: pd.DataFrame) -> str:
    def clean_clause(text: str, max_len: int = 140) -> str:
        text = text.strip()
        if len(text) <= max_len:
            return text
        truncated = text[:max_len].rsplit(" ", 1)[0]
        return truncated + "…"

    long_summary = info.get("longBusinessSummary") or ""
    first_clause = ""
    if long_summary:
        first_clause = re.split(r'[.;]', long_summary, 1)[0].strip()

    name = info.get("longName") or info.get("shortName") or "The company"
    industry = info.get("industry") or info.get("sector") or "technology"
    description = clean_clause(first_clause) if first_clause else f"{name} operates in the {industry.lower()} space with global reach."

    monetization = "It monetizes through diversified platforms and services across enterprise and consumer channels."
    keywords = {
        "Semiconductor": "It monetizes primarily through high-performance chips, accelerator boards, and software subscriptions.",
        "Software": "It monetizes via subscription software, enterprise licenses, and cloud-delivered services.",
        "Energy": "It monetizes through power generation assets, infrastructure services, and long-term contracts.",
        "Financial": "It monetizes through transaction services, data solutions, and asset-light advisory products.",
    }
    for key, sentence in keywords.items():
        if key.lower() in industry.lower():
            monetization = sentence
            break

    revenue_series = safe_series(fin_df, "Total Revenue")
    latest_revenue = format_currency(revenue_series.iloc[-1]) if not revenue_series.empty else None
    revenue_growth = None
    if len(revenue_series) >= 5:
        revenue_growth = revenue_series.pct_change().iloc[-1]

    if latest_revenue and revenue_growth is not None:
        outlook = f"Recent quarterly revenue was roughly {latest_revenue}, growing {revenue_growth:.1%} vs. the prior period as AI demand scales."
    elif latest_revenue:
        outlook = f"Recent quarterly revenue was roughly {latest_revenue}, supported by accelerating infrastructure and software demand."
    else:
        outlook = "Recent revenue trends are not disclosed in the available filings."

    return " ".join([description, monetization, outlook])


@dataclass
class ComponentScore:
    score: Optional[float]
    summary: str
    inputs: Dict[str, float]
    notes: List[str]


def clamp(value: float, lower: float = 0.0, upper: float = 10.0) -> float:
    return max(lower, min(value, upper))


def score_from_thresholds(value: float, thresholds: List[Tuple[float, float]]) -> float:
    """
    Map a metric value to a score using ordered thresholds.
    thresholds: list of (threshold_value, score). Sorted ascending.
    """
    for threshold, score in thresholds:
        if value <= threshold:
            return score
    return thresholds[-1][1]


def load_financials_df(ticker: str) -> pd.DataFrame:
    path = FINANCIALS_DIR / f"{ticker}_financials.json"
    if not path.exists():
        raise ScoreComputationError(f"Financials file not found for {ticker}")
    df = pd.read_json(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df.set_index("date")


def load_earnings_df(ticker: str) -> pd.DataFrame:
    path = EARNINGS_DIR / f"{ticker}_quarterly_earnings.json"
    if not path.exists():
        raise ScoreComputationError(f"Earnings file not found for {ticker}")
    df = pd.read_json(path)
    df["period"] = pd.to_datetime(df["period"])
    df = df.sort_values("period")
    return df.set_index("period")


def load_price_history(ticker: str) -> pd.DataFrame:
    path = PRICES_DIR / f"{ticker}_prices.csv"
    if not path.exists():
        raise ScoreComputationError(f"Price history not found for {ticker}")
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df


def calculate_cagr(series: pd.Series) -> Optional[float]:
    if series.empty or len(series) < 2:
        return None
    start_value = series.iloc[0]
    end_value = series.iloc[-1]
    if start_value <= 0 or end_value <= 0:
        return None
    periods = len(series) - 1
    if periods == 0:
        return None
    cagr = (end_value / start_value) ** (1 / periods) - 1
    return cagr


def safe_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=float)
    return df[column].dropna()


def compute_business_score(ticker: str, fin_df: pd.DataFrame, info: dict) -> ComponentScore:
    revenue_series = safe_series(fin_df, "Total Revenue")
    if revenue_series.empty:
        raise ScoreComputationError(f"Total Revenue series missing for {ticker}")

    gross_profit_series = safe_series(fin_df, "Gross Profit")
    gross_margin_series = (gross_profit_series / revenue_series).dropna() if not gross_profit_series.empty else pd.Series(dtype=float)

    rnd_series_raw = safe_series(fin_df, "Research And Development")
    rnd_series = (rnd_series_raw / revenue_series).dropna() if not rnd_series_raw.empty else pd.Series(dtype=float)

    revenue_cagr = calculate_cagr(revenue_series[-4:])
    gross_margin_latest = gross_margin_series.iloc[-1] if not gross_margin_series.empty else None
    rnd_intensity_latest = rnd_series.iloc[-1] if not rnd_series.empty else None

    industry = info.get("industry") or ""
    sector = info.get("sector") or ""

    industry_score = SECTOR_GROWTH_BONUS.get(industry, SECTOR_GROWTH_BONUS.get(sector, 6.0))
    moat_score_components = []

    notes = []

    if revenue_cagr is not None:
        revenue_score = score_from_thresholds(
            revenue_cagr,
            [(-0.05, 2), (0.0, 4), (0.05, 6), (0.15, 8), (0.30, 10)],
        )
        moat_score_components.append(revenue_score)
    else:
        notes.append("Insufficient data to compute revenue CAGR.")

    if gross_margin_latest is not None:
        gm_score = score_from_thresholds(
            gross_margin_latest,
            [(0.2, 3), (0.35, 5), (0.45, 7), (0.55, 9), (0.65, 10)],
        )
        moat_score_components.append(gm_score)
    else:
        notes.append("Missing gross margin data.")

    if rnd_intensity_latest is not None:
        rnd_score = score_from_thresholds(
            rnd_intensity_latest,
            [(0.05, 4), (0.1, 6), (0.15, 8), (0.20, 9), (0.30, 10)],
        )
        moat_score_components.append(rnd_score)
    else:
        notes.append("Research & Development intensity unavailable.")

    if not moat_score_components:
        moat_score = 5.0
    else:
        moat_score = mean(moat_score_components)

    combined = mean([industry_score, moat_score])

    summary = (
        f"Operating in {industry or sector}, revenue CAGR of "
        f"{revenue_cagr:.1%} and gross margin of {gross_margin_latest:.1%} signal a strong moat."
        if revenue_cagr is not None and gross_margin_latest is not None
        else "Industry fundamentals are favorable; additional moat data is limited."
    )

    return ComponentScore(
        score=clamp(combined),
        summary=summary,
        inputs={
            "revenue_cagr": revenue_cagr if revenue_cagr is not None else np.nan,
            "gross_margin": gross_margin_latest if gross_margin_latest is not None else np.nan,
            "rnd_intensity": rnd_intensity_latest if rnd_intensity_latest is not None else np.nan,
            "industry_score": industry_score,
        },
        notes=notes,
    )


def compute_financial_score(fin_df: pd.DataFrame, info: dict) -> ComponentScore:
    revenue_series = safe_series(fin_df, "Total Revenue")
    net_income_series = safe_series(fin_df, "Net Income")
    if revenue_series.empty or net_income_series.empty:
        raise ScoreComputationError("Insufficient revenue/net income data for financial scoring.")

    revenue_cagr = calculate_cagr(revenue_series[-4:])
    net_margin_series = (net_income_series / revenue_series).dropna()
    net_margin_latest = net_margin_series.iloc[-1] if not net_margin_series.empty else None

    free_cashflow = info.get("freeCashflow")
    total_debt = info.get("totalDebt")
    debt_to_equity = info.get("debtToEquity")
    pe_ratio = info.get("trailingPE")
    ps_ratio = info.get("priceToSalesTrailing12Months")

    notes: List[str] = []
    component_scores: List[float] = []

    if revenue_cagr is not None:
        component_scores.append(
            score_from_thresholds(
                revenue_cagr,
                [(-0.05, 2), (0.0, 4), (0.05, 6), (0.15, 8), (0.30, 10)],
            )
        )

    if net_margin_latest is not None:
        component_scores.append(
            score_from_thresholds(
                net_margin_latest,
                [(0.0, 3), (0.05, 5), (0.10, 6.5), (0.20, 8.5), (0.30, 10)],
            )
        )

    if free_cashflow and total_debt and total_debt > 0:
        fcf_coverage = free_cashflow / total_debt
        component_scores.append(
            score_from_thresholds(
                fcf_coverage,
                [(0.2, 3), (0.5, 5), (1.0, 7), (1.5, 9), (2.0, 10)],
            )
        )
    else:
        notes.append("Free cash flow or debt data missing; assuming neutral coverage.")
        component_scores.append(5.5)

    if debt_to_equity is not None and debt_to_equity > 0:
        component_scores.append(
            score_from_thresholds(
                debt_to_equity,
                [(50, 9.5), (100, 8.5), (200, 6.5), (300, 4.5), (500, 3.0), (800, 2.0)],
            )
        )
    else:
        notes.append("Debt-to-equity not available; assuming neutral leverage.")
        component_scores.append(6.0)

    if pe_ratio:
        component_scores.append(
            score_from_thresholds(
                pe_ratio,
                [(15, 9.5), (25, 8.5), (40, 7.0), (55, 5.0), (80, 3.5), (110, 2.0)],
            )
        )

    if ps_ratio:
        component_scores.append(
            score_from_thresholds(
                ps_ratio,
                [(2, 9.0), (4, 8.0), (6, 7.0), (8, 6.0), (12, 4.5), (18, 3.0)],
            )
        )

    score = clamp(mean(component_scores)) if component_scores else None

    summary = (
        f"Revenue CAGR {revenue_cagr:.1%}, net margin {net_margin_latest:.1%}, "
        f"with free cash flow coverage {free_cashflow / total_debt:.2f}x."
        if revenue_cagr is not None and net_margin_latest is not None and free_cashflow and total_debt
        else "Financial metrics indicate solid fundamentals with some data gaps."
    )

    return ComponentScore(
        score=score,
        summary=summary,
        inputs={
            "revenue_cagr": revenue_cagr if revenue_cagr is not None else np.nan,
            "net_margin": net_margin_latest if net_margin_latest is not None else np.nan,
            "free_cashflow": free_cashflow if free_cashflow is not None else np.nan,
            "total_debt": total_debt if total_debt is not None else np.nan,
            "debt_to_equity": debt_to_equity if debt_to_equity is not None else np.nan,
            "trailing_pe": pe_ratio if pe_ratio is not None else np.nan,
            "price_to_sales": ps_ratio if ps_ratio is not None else np.nan,
        },
        notes=notes,
    )


def compute_event_sentiment_score(ticker: str, news_items: List[dict], reddit_stats: Optional[dict]) -> ComponentScore:
    sentiments = []
    timeline = []
    now = datetime.now(timezone.utc)

    for item in news_items or []:
        content = item.get("content") or {}
        title = content.get("title") or item.get("title") or "Event"
        summary = content.get("summary") or content.get("description") or title
        provider = (content.get("provider") or {}).get("displayName") or (item.get("provider") or {}).get("displayName") or "Unknown"
        link = (
            item.get("link")
            or (item.get("canonicalUrl") or {}).get("url")
            or (item.get("clickThroughUrl") or {}).get("url")
            or (content.get("canonicalUrl") or {}).get("url")
        )
        provider_time = (
            item.get("providerPublishTime")
            or content.get("pubDate")
        )
        if isinstance(provider_time, (int, float)):
            published_at = datetime.fromtimestamp(provider_time, tz=timezone.utc)
        elif isinstance(provider_time, str) and provider_time:
            try:
                published_at = datetime.fromisoformat(provider_time.replace("Z", "+00:00"))
            except ValueError:
                published_at = now
        else:
            published_at = now
        age_days = (now - published_at).days
        sentiment = sentiment_analyzer.polarity_scores(title or summary)
        weight = max(0.2, 1 - (age_days / 30))
        sentiments.append(sentiment["compound"] * weight)
        timeline.append(
            {
                "title": title,
                "source": provider,
                "published_at": published_at.isoformat(),
                "link": link,
                "sentiment": sentiment["compound"],
                "weight": weight,
            }
        )

    notes = []

    news_score = None
    coverage = 0.0
    avg_sentiment = 0.0
    if sentiments:
        coverage = min(len(news_items), 10) / 10
        avg_sentiment = sum(sentiments) / len(sentiments)
        base = score_from_thresholds(
            avg_sentiment,
            [(-0.6, 1), (-0.2, 4), (0.0, 5.5), (0.2, 7.5), (0.45, 9.5)],
        )
        news_score = clamp(base * (0.5 + 0.5 * coverage))
    else:
        notes.append("No recent mainstream news detected.")

    reddit_score = None
    reddit_ratio = None
    reddit_total = 0
    if reddit_stats and reddit_stats.get("total_posts"):
        reddit_total = reddit_stats.get("total_posts", 0)
        bullish = reddit_stats.get("bullish_posts", 0)
        bearish = reddit_stats.get("bearish_posts", 0)
        reddit_ratio = (bullish - bearish) / max(reddit_total, 1)
        reddit_base = score_from_thresholds(
            reddit_ratio,
            [(-0.6, 2), (-0.2, 4), (0.0, 6), (0.2, 8), (0.4, 9.5)],
        )
        reddit_coverage = min(reddit_total / 15, 1.0)
        reddit_score = clamp(reddit_base * (0.4 + 0.6 * reddit_coverage))
    else:
        notes.append("No recent Reddit sentiment captured.")

    combined_scores = [s for s in [news_score, reddit_score] if s is not None]
    final_score = clamp(mean(combined_scores)) if combined_scores else 5.0

    summary_parts = []
    if news_score is not None:
        summary_parts.append(f"News sentiment {avg_sentiment:.2f} across {len(news_items)} items.")
    if reddit_score is not None:
        summary_parts.append(f"Reddit activity {reddit_total} posts with bullish ratio {reddit_ratio:.2f}.")
    if not summary_parts:
        summary_parts.append("Sentiment data limited; neutral baseline applied.")

    summary = " ".join(summary_parts)

    return ComponentScore(
        score=final_score,
        summary=summary,
        inputs={
            "average_sentiment": avg_sentiment,
            "coverage": coverage,
            "reddit_ratio": reddit_ratio if reddit_ratio is not None else np.nan,
            "reddit_posts": reddit_total,
        },
        notes=notes,
    )


def compute_critical_path_score(ticker: str, info: dict) -> ComponentScore:
    base = CRITICAL_PATH_MAP.get(ticker.upper(), 5.0)
    industry = info.get("industry", "")
    summary = f"{ticker} operates in {industry or info.get('sector', 'its sector')}, weighted criticality score {base:.1f}."
    return ComponentScore(
        score=base,
        summary=summary,
        inputs={"industry": industry, "sector": info.get("sector")},
        notes=[],
    )


def compute_leadership_score(ticker: str, info: dict) -> ComponentScore:
    officers = info.get("companyOfficers") or []
    ceo = next((o for o in officers if "CEO" in (o.get("title") or "").upper()), None)
    profile = load_ceo_profile(ticker)
    tenure_years: Optional[float] = None
    education = None
    linkedin_url = None
    reputation_bonus = 0.0

    if profile:
        education = profile.get("education")
        linkedin_url = profile.get("linkedin_url")
        start_date = profile.get("start_date")
        if isinstance(start_date, str) and start_date and start_date.lower() != "not found":
            start_dt = pd.to_datetime(start_date, errors="coerce")
            if pd.notnull(start_dt):
                tenure_years = (datetime.utcnow() - start_dt.to_pydatetime()).days / 365.0
        tenure_duration = profile.get("tenure_duration")
        if tenure_years is None and isinstance(tenure_duration, str) and tenure_duration.lower() != "not found":
            match = re.search(r"(\d+)", tenure_duration)
            if match:
                tenure_years = float(match.group(1))
        try:
            num_highlights = float(profile.get("num_highlights", 0))
        except (TypeError, ValueError):
            num_highlights = 0.0
        if num_highlights > 0:
            reputation_bonus += 0.5

    notes: List[str] = []
    score_components: List[float] = []

    if ceo:
        title = ceo.get("title", "")
        total_pay = ceo.get("totalPay")
        age = ceo.get("age")
        if "CO-FOUNDER" in title.upper():
            score_components.append(9.0)
        if total_pay and total_pay > 5_000_000:
            score_components.append(8.5)
        if age and age >= 55:
            score_components.append(7.5)
        if total_pay and total_pay > 10_000_000:
            score_components.append(9.5)
    else:
        notes.append("CEO not listed in officer roster; using neutral baseline.")

    if tenure_years is not None:
        score_components.append(
            score_from_thresholds(
                tenure_years,
                [(2, 6.0), (4, 7.0), (6, 8.0), (10, 9.0), (15, 9.5)],
            )
        )
    elif ceo and ceo.get("maxAge"):
        notes.append("Tenure duration not available; partial leadership score applied.")

    if len(officers) >= 5:
        score_components.append(7.5)
    if any("Chief Scientist" in (o.get("title") or "") for o in officers):
        score_components.append(7.5)

    if reputation_bonus:
        score_components.append(7.0 + reputation_bonus)

    score = clamp(mean(score_components) if score_components else 5.5)

    summary_parts = []
    if ceo:
        summary_parts.append(
            f"Led by {ceo.get('name')} ({ceo.get('title')})."
        )
        if ceo.get("totalPay"):
            summary_parts.append(f"Compensation ${ceo.get('totalPay'):,}.")
    if tenure_years:
        summary_parts.append(f"Approx. tenure {tenure_years:.1f} years.")
    if education and isinstance(education, str) and education.lower() != "not found":
        summary_parts.append(f"Education: {education}.")
    if not summary_parts:
        summary_parts.append("Leadership data limited; defaulting to neutral score.")

    summary = " ".join(summary_parts)

    return ComponentScore(
        score=score,
        summary=summary,
        inputs={
            "officer_count": len(officers),
            "ceo_tenure_years": tenure_years if tenure_years is not None else np.nan,
        },
        notes=notes,
    )


def compute_earnings_score(earnings_df: pd.DataFrame) -> ComponentScore:
    if earnings_df.empty:
        return ComponentScore(
            score=5.0,
            summary="No quarterly earnings data available; treating as neutral.",
            inputs={},
            notes=["Earnings dataset empty."],
        )

    eps = earnings_df["eps"].dropna()
    revenue = earnings_df["revenue"].dropna()
    earnings = earnings_df["earnings"].dropna()

    if len(eps) < 2:
        return ComponentScore(
            score=5.0,
            summary="Insufficient EPS history for scoring; neutral baseline applied.",
            inputs={"eps_points": len(eps)},
            notes=[],
        )

    eps_growth = eps.pct_change().dropna()
    revenue_growth = revenue.pct_change().dropna()

    eps_consistency = 1 / (eps_growth.std() + 1e-6)
    revenue_consistency = 1 / (revenue_growth.std() + 1e-6)

    eps_trend = eps_growth.mean()
    revenue_trend = revenue_growth.mean()

    eps_score = score_from_thresholds(
        eps_trend,
        [(-0.1, 2), (0.0, 4), (0.05, 6), (0.15, 8.5), (0.30, 10)],
    )
    rev_score = score_from_thresholds(
        revenue_trend,
        [(-0.05, 2), (0.0, 4), (0.05, 6.5), (0.15, 8.5), (0.30, 10)],
    )
    stability_score = score_from_thresholds(
        eps_consistency,
        [(2, 4), (5, 6), (10, 8), (15, 9), (25, 10)],
    )

    combined = clamp(mean([eps_score, rev_score, stability_score]))

    summary = (
        f"EPS trend {eps_trend:.1%}, revenue trend {revenue_trend:.1%}, "
        f"with consistency score {stability_score:.1f}."
    )

    return ComponentScore(
        score=combined,
        summary=summary,
        inputs={
            "eps_trend": eps_trend,
            "revenue_trend": revenue_trend,
            "consistency_metric": eps_consistency,
        },
        notes=[],
    )


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_technical_score(price_df: pd.DataFrame) -> ComponentScore:
    if price_df.empty:
        return ComponentScore(
            score=5.0,
            summary="Price history unavailable; technical stance neutral.",
            inputs={},
            notes=[],
        )

    closes = price_df["Close"]
    rsi_series = compute_rsi(closes).dropna()
    if rsi_series.empty:
        return ComponentScore(
            score=5.0,
            summary="Insufficient price data for RSI; technical score neutral.",
            inputs={},
            notes=[],
        )

    latest_rsi = rsi_series.iloc[-1]

    ema12 = ema(closes, 12)
    ema26 = ema(closes, 26)
    macd = ema12 - ema26
    signal = ema(macd, 9)
    macd_cross = macd.iloc[-1] - signal.iloc[-1]

    ma50 = closes.rolling(window=50).mean()
    ma200 = closes.rolling(window=200).mean()

    latest_close = closes.iloc[-1]
    ma50_latest = ma50.iloc[-1] if not ma50.dropna().empty else None
    ma200_latest = ma200.iloc[-1] if not ma200.dropna().empty else None

    score = 5.0
    notes: List[str] = []

    if latest_rsi < 30:
        score += 3
        notes.append("RSI indicates oversold conditions.")
    elif latest_rsi > 70:
        score -= 3
        notes.append("RSI indicates overbought conditions.")

    if macd_cross > 0:
        score += 2
        notes.append("MACD bullish crossover detected.")
    elif macd_cross < 0:
        score -= 2
        notes.append("MACD bearish crossover detected.")

    if ma50_latest and ma200_latest:
        if ma50_latest > ma200_latest:
            score += 3
            notes.append("Golden cross (MA50 > MA200).")
        else:
            score -= 2
            notes.append("Bearish moving average alignment.")

    score = clamp(score)

    summary = (
        f"RSI {latest_rsi:.1f}, MACD differential {macd_cross:.2f}, "
        f"closing price ${latest_close:,.2f}."
    )

    return ComponentScore(
        score=score,
        summary=summary,
        inputs={
            "rsi": latest_rsi,
            "macd_delta": macd_cross,
            "close": latest_close,
            "ma50": ma50_latest if ma50_latest is not None else np.nan,
            "ma200": ma200_latest if ma200_latest is not None else np.nan,
        },
        notes=notes,
    )


def compute_overall_score(component_scores: Dict[str, ComponentScore]) -> Dict[str, float]:
    weights = {
        "business": 0.20,
        "financial": 0.25,
        "sentiment": 0.15,
        "critical": 0.10,
        "leadership": 0.10,
        "earnings": 0.10,
        "technical": 0.10,
    }

    weighted_scores = []
    applied_weights = []
    available_scores = []

    for key, comp in component_scores.items():
        if comp.score is not None:
            weight = weights.get(key, 0.0)
            weighted_scores.append(comp.score * weight)
            applied_weights.append(weight)
            available_scores.append(comp.score)

    total_weight = sum(applied_weights) or 1.0
    aggregate = sum(weighted_scores) / total_weight

    if available_scores:
        variance = pvariance(available_scores)
        confidence = clamp(1 - variance / 36, 0.4, 0.95)
    else:
        confidence = 0.5

    if aggregate >= 8:
        recommendation = "Strong Buy"
        hold_duration = "Long-term (12-24 months horizon)"
    elif aggregate >= 7:
        recommendation = "Buy"
        hold_duration = "Long-term (12-18 months horizon)"
    elif aggregate >= 4:
        recommendation = "Hold"
        hold_duration = "Medium-term (6-12 months horizon)"
    else:
        recommendation = "Sell"
        hold_duration = "Reevaluate position in the short term (<6 months)"

    return {
        "score": round(aggregate, 2),
        "confidence": round(confidence * 100, 1),
        "recommendation": recommendation,
        "hold_duration": hold_duration,
    }


def build_event_timeline(news_items: List[dict], max_items: int = 5) -> List[dict]:
    timeline = []
    for item in news_items[:max_items]:
        content = item.get("content") or {}
        title = content.get("title") or item.get("title") or "Event"
        provider = (content.get("provider") or {}).get("displayName") or (item.get("provider") or {}).get("displayName") or "Unknown"
        link = (
            item.get("link")
            or (item.get("canonicalUrl") or {}).get("url")
            or (item.get("clickThroughUrl") or {}).get("url")
            or (content.get("canonicalUrl") or {}).get("url")
        )
        provider_time = item.get("providerPublishTime") or content.get("pubDate")
        if isinstance(provider_time, (int, float)):
            published_at = datetime.fromtimestamp(provider_time, tz=timezone.utc)
        elif isinstance(provider_time, str) and provider_time:
            try:
                published_at = datetime.fromisoformat(provider_time.replace("Z", "+00:00"))
            except ValueError:
                published_at = datetime.utcnow().replace(tzinfo=timezone.utc)
        else:
            published_at = datetime.utcnow().replace(tzinfo=timezone.utc)

        sentiment = sentiment_analyzer.polarity_scores(title)
        timeline.append(
            {
                "title": title,
                "source": provider,
                "link": link,
                "published_at": published_at.isoformat(),
                "sentiment": sentiment["compound"],
            }
        )
    return timeline


def compute_company_scores(ticker: str) -> Dict[str, object]:
    ticker = ticker.upper()
    
    # Load data with error handling
    try:
        fin_df = load_financials_df(ticker)
    except ScoreComputationError:
        fin_df = pd.DataFrame()  # Empty DataFrame as fallback
    
    try:
        earnings_df = load_earnings_df(ticker)
    except ScoreComputationError:
        earnings_df = pd.DataFrame()  # Empty DataFrame as fallback
    
    try:
        price_df = load_price_history(ticker)
    except ScoreComputationError:
        price_df = pd.DataFrame()  # Empty DataFrame as fallback

    # Get Yahoo Finance data with error handling
    try:
        instrument = yf.Ticker(ticker)
        info = instrument.info or {}
        news_items = instrument.news or []
    except Exception:
        info = {}
        news_items = []
    
    try:
        reddit_summary = load_latest_reddit_summary().get(ticker)
    except Exception:
        reddit_summary = None

    # Compute scores with individual error handling
    try:
        business = compute_business_score(ticker, fin_df, info)
    except Exception as e:
        business = ComponentScore(
            score=None,
            summary=f"Business score calculation failed: {str(e)}",
            inputs={},
            notes=[f"Error: {str(e)}"]
        )
    
    try:
        financial = compute_financial_score(fin_df, info)
    except Exception as e:
        financial = ComponentScore(
            score=None,
            summary=f"Financial score calculation failed: {str(e)}",
            inputs={},
            notes=[f"Error: {str(e)}"]
        )
    
    try:
        sentiment = compute_event_sentiment_score(ticker, news_items, reddit_summary)
    except Exception as e:
        sentiment = ComponentScore(
            score=None,
            summary=f"Sentiment score calculation failed: {str(e)}",
            inputs={},
            notes=[f"Error: {str(e)}"]
        )
    
    try:
        critical = compute_critical_path_score(ticker, info)
    except Exception as e:
        critical = ComponentScore(
            score=None,
            summary=f"Critical score calculation failed: {str(e)}",
            inputs={},
            notes=[f"Error: {str(e)}"]
        )
    
    try:
        leadership = compute_leadership_score(ticker, info)
    except Exception as e:
        leadership = ComponentScore(
            score=None,
            summary=f"Leadership score calculation failed: {str(e)}",
            inputs={},
            notes=[f"Error: {str(e)}"]
        )
    
    try:
        earnings = compute_earnings_score(earnings_df)
    except Exception as e:
        earnings = ComponentScore(
            score=None,
            summary=f"Earnings score calculation failed: {str(e)}",
            inputs={},
            notes=[f"Error: {str(e)}"]
        )
    
    try:
        technical = compute_technical_score(price_df)
    except Exception as e:
        technical = ComponentScore(
            score=None,
            summary=f"Technical score calculation failed: {str(e)}",
            inputs={},
            notes=[f"Error: {str(e)}"]
        )

    component_scores = {
        "business": business,
        "financial": financial,
        "sentiment": sentiment,
        "critical": critical,
        "leadership": leadership,
        "earnings": earnings,
        "technical": technical,
    }

    overall = compute_overall_score(component_scores)

    company_profile = {
        "ticker": ticker,
        "name": info.get("longName") or info.get("shortName") or ticker,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "summary": None,
        "website": info.get("website"),
        "country": info.get("country"),
        "employees": info.get("fullTimeEmployees"),
        "founded": info.get("foundedYear") or extract_year_from_summary(info.get("longBusinessSummary")),
        "logo_url": info.get("logo_url"),
    }

    company_profile["summary"] = summarize_company(info, fin_df)

    return {
        "generated_at": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
        "company": company_profile,
        "scores": {
            key: {
                "score": round(comp.score, 2) if comp.score is not None else None,
                "summary": comp.summary,
                "inputs": {
                    k: (
                        float(v)
                        if isinstance(v, (int, float, np.floating, np.integer)) and v == v
                        else (v if isinstance(v, str) else None)
                    )
                    for k, v in comp.inputs.items()
                },
                "notes": comp.notes,
            }
            for key, comp in component_scores.items()
        },
        "overall": overall,
        "event_timeline": build_event_timeline(news_items),
    }
