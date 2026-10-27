"""
CANSLIM Thematic Concentration & Contagion Radar Engine (Pillar 1)
==================================================================
Eliminates institutional blind spots by tracking sector-wide fund liquidations,
thematic supply rollover, and O'Neil 50% Rule pack distribution.

100% Deterministic 3-Tier Classification Architecture:
  Tier 1: Master Database Resolution (industry_map.json + SQLite fundamentals_cache)
  Tier 2: Exchange / yfinance API Dynamic Fallback & Persistence
  Tier 3: Immutable Sub-Industry -> Thematic Super-Cluster Mapping Engine
"""

import os
import json
import sqlite3
import numpy as np
import pandas as pd

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "kush_tracker.db")
INDUSTRY_MAP_PATH = os.path.join(BASE_DIR, "industry_map.json")
BSE_MAP_PATH = os.path.join(BASE_DIR, "bse_mapping.json")

# In-memory caches to ensure fast repeated execution
_CACHED_INDUSTRY_MAP = None
_CACHED_BSE_REV_MAP = None
_CACHED_DB_FUNDAMENTALS = None

# =============================================================================
# TIER 3: IMMUTABLE SUB-INDUSTRY -> CANONICAL THEMATIC SUPER-CLUSTER ENGINE
# =============================================================================
SUB_INDUSTRY_TO_THEME = {
    # 🏥 Pharmaceuticals & Healthcare
    "Drug Manufacturers - Specialty & Generic": "🏥 Pharmaceuticals & Healthcare",
    "Drug Manufacturers - General":             "🏥 Pharmaceuticals & Healthcare",
    "Medical Care Facilities":                  "🏥 Pharmaceuticals & Healthcare",
    "Biotechnology":                            "🏥 Pharmaceuticals & Healthcare",
    "Medical Instruments & Supplies":           "🏥 Pharmaceuticals & Healthcare",
    "Diagnostics & Research":                   "🏥 Pharmaceuticals & Healthcare",
    "Healthcare Plans":                         "🏥 Pharmaceuticals & Healthcare",
    "Medical Distribution":                     "🏥 Pharmaceuticals & Healthcare",

    # 🏭 Automotive & Mobility
    "Auto Manufacturers":                       "🏭 Automotive & Mobility",
    "Auto Parts":                               "🏭 Automotive & Mobility",
    "Auto & Truck Dealerships":                 "🏭 Automotive & Mobility",
    "Recreational Vehicles":                    "🏭 Automotive & Mobility",

    # 🧪 Specialty Chemicals & Materials
    "Specialty Chemicals":                      "🧪 Specialty Chemicals & Materials",
    "Chemicals":                                "🧪 Specialty Chemicals & Materials",
    "Agricultural Inputs":                      "🧪 Specialty Chemicals & Materials",

    # 💎 Luxury Retail & Jewelry
    "Luxury Goods":                             "💎 Luxury Retail & Jewelry",
    "Apparel Retail":                           "💎 Luxury Retail & Jewelry",
    "Footwear & Accessories":                   "💎 Luxury Retail & Jewelry",
    "Apparel Manufacturing":                    "💎 Luxury Retail & Jewelry",
    "Textile Manufacturing":                    "💎 Luxury Retail & Jewelry",

    # 🔌 Electrical Equipment & Cables
    "Electrical Equipment & Parts":             "🔌 Electrical Equipment & Cables",
    "Electronic Components":                    "🔌 Electronics Hardware & Components",
    "Consumer Electronics":                     "🔌 Electronics Hardware & Components",
    "Electronics & Computer Distribution":      "🔌 Electronics Hardware & Distribution",

    # 🛡️ Defense & Aerospace
    "Aerospace & Defense":                      "🛡️ Defense & Aerospace",

    # ☀️ Solar & Renewable Energy
    "Solar":                                    "☀️ Solar & Renewable Energy",
    "Utilities - Renewable":                    "☀️ Solar & Renewable Energy",

    # ⚡ Power T&D & Utilities
    "Utilities - Regulated Electric":           "⚡ Power T&D & Utilities",
    "Utilities - Independent Power Producers":  "⚡ Power T&D & Utilities",
    "Utilities - Diversified":                  "⚡ Power T&D & Utilities",

    # ⚙️ Industrial Machinery & Engineering
    "Specialty Industrial Machinery":           "⚙️ Industrial Machinery & Engineering",
    "Metal Fabrication":                        "⚙️ Industrial Machinery & Engineering",
    "Pollution & Treatment Controls":           "⚙️ Industrial Machinery & Engineering",
    "Industrial Distribution":                  "⚙️ Industrial Machinery & Engineering",
    "Tools & Accessories":                      "⚙️ Industrial Machinery & Engineering",

    # 🏗️ Engineering & Infrastructure
    "Engineering & Construction":               "🏗️ Engineering & Infrastructure",
    "Infrastructure Operations":                "🏗️ Engineering & Infrastructure",

    # 🏗️ Building Products & Materials
    "Building Products & Equipment":            "🏗️ Building Products & Materials",
    "Building Materials":                       "🏗️ Building Products & Materials",
    "Cement":                                   "🏗️ Building Products & Materials",

    # 📦 Logistics & Supply Chain
    "Integrated Freight & Logistics":           "📦 Logistics & Supply Chain",
    "Marine Shipping":                          "📦 Logistics & Supply Chain",
    "Airlines":                                 "📦 Logistics & Supply Chain",
    "Railroads":                                "📦 Logistics & Supply Chain",
    "Airports & Air Services":                  "📦 Logistics & Supply Chain",

    # ⛏️ Metals & Mining
    "Copper":                                   "⛏️ Metals & Mining",
    "Other Industrial Metals & Mining":         "⛏️ Metals & Mining",
    "Steel":                                    "⛏️ Metals & Mining",
    "Aluminum":                                 "⛏️ Metals & Mining",
    "Other Precious Metals & Mining":           "⛏️ Metals & Mining",
    "Gold":                                     "⛏️ Metals & Mining",

    # 📡 Telecom & Communications
    "Communication Equipment":                  "📡 Telecom & Communications",
    "Telecom Services":                         "📡 Telecom & Communications",

    # 💻 Software & Technology
    "Software - Application":                   "💻 Software & Technology",
    "Software - Infrastructure":                "💻 Software & Technology",
    "Information Technology Services":          "💻 Software & Technology",

    # 📈 Capital Markets & Wealth Platforms
    "Capital Markets":                          "📈 Capital Markets & Platforms",
    "Asset Management":                         "📈 Capital Markets & Platforms",
    "Financial Data & Stock Exchanges":         "📈 Capital Markets & Platforms",
    "Insurance Brokers":                        "📈 Capital Markets & Platforms",

    # 🏦 Banking & Financial Services
    "Banks - Regional":                         "🏦 Banking & Financial Services",
    "Banks - Diversified":                      "🏦 Banking & Financial Services",
    "Credit Services":                          "🏦 Banking & Financial Services",

    # 🏨 Hotels & Premium Hospitality
    "Lodging":                                  "🏨 Hotels & Premium Hospitality",
    "Resorts & Casinos":                        "🏨 Hotels & Premium Hospitality",
    "Restaurants":                              "🏨 Hotels & Premium Hospitality",

    # 🌾 FMCG & Consumer Staples
    "Packaged Foods":                           "🌾 FMCG & Consumer Staples",
    "Beverages - Non-Alcoholic":                "🌾 FMCG & Consumer Staples",
    "Beverages - Wineries & Distilleries":      "🌾 FMCG & Consumer Staples",
    "Confectioners":                            "🌾 FMCG & Consumer Staples",
    "Household & Personal Products":            "🌾 FMCG & Consumer Staples",

    # 💰 Cash & Commodities
    "Gold ETF":                                 "💰 Cash & Commodities",
    "Government Securities Fund":               "💰 Cash & Commodities",
    "Cash / Liquid":                            "💰 Cash & Commodities",
}

# Fallback known Indian ETFs
KNOWN_ETF_THEMES = {
    "GOLDBEES": ("Gold / Commodity", "Gold ETF", "💰 Cash & Commodities"),
    "LIQUIDBEES": ("Cash / Liquid", "Government Securities Fund", "💰 Cash & Commodities"),
    "LIQUIDCASE": ("Cash / Liquid", "Government Securities Fund", "💰 Cash & Commodities"),
    "LIQUIDETF": ("Cash / Liquid", "Government Securities Fund", "💰 Cash & Commodities"),
}


def load_bse_reverse_map():
    """Load and cache BSE numerical code reverse mapping."""
    global _CACHED_BSE_REV_MAP
    if _CACHED_BSE_REV_MAP is not None:
        return _CACHED_BSE_REV_MAP

    rev_map = {}
    if os.path.exists(BSE_MAP_PATH):
        try:
            with open(BSE_MAP_PATH, "r", encoding="utf-8") as f:
                bmap = json.load(f)
                rev_map = {str(v): k for k, v in bmap.items()}
        except Exception:
            pass
    _CACHED_BSE_REV_MAP = rev_map
    return _CACHED_BSE_REV_MAP


def load_canonical_industry_map():
    """Load and cache the 3,542 pre-cached industry mappings."""
    global _CACHED_INDUSTRY_MAP
    if _CACHED_INDUSTRY_MAP is not None:
        return _CACHED_INDUSTRY_MAP

    imap = {}
    if os.path.exists(INDUSTRY_MAP_PATH):
        try:
            with open(INDUSTRY_MAP_PATH, "r", encoding="utf-8") as f:
                imap = json.load(f)
        except Exception:
            pass
    _CACHED_INDUSTRY_MAP = imap
    return _CACHED_INDUSTRY_MAP


def load_db_fundamentals_cache():
    """Load and cache sqlite fundamentals cache."""
    global _CACHED_DB_FUNDAMENTALS
    if _CACHED_DB_FUNDAMENTALS is not None:
        return _CACHED_DB_FUNDAMENTALS

    fcache = {}
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("SELECT ticker, industry FROM fundamentals_cache WHERE industry IS NOT NULL AND industry != ''")
            for ticker, ind in c.fetchall():
                fcache[ticker] = ind
            conn.close()
        except Exception:
            pass
    _CACHED_DB_FUNDAMENTALS = fcache
    return _CACHED_DB_FUNDAMENTALS


def persist_industry_to_cache(ticker, industry):
    """Persist newly fetched industry to SQLite fundamentals_cache."""
    if not industry or industry in ("Other / ETF", "Unknown"):
        return
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO fundamentals_cache (ticker, industry, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """, (ticker, industry))
        conn.commit()
        conn.close()
        # Update in-memory cache
        global _CACHED_DB_FUNDAMENTALS
        if _CACHED_DB_FUNDAMENTALS is not None:
            _CACHED_DB_FUNDAMENTALS[ticker] = industry
    except Exception:
        pass


def resolve_symbol(ticker, bse_reverse_map=None):
    """
    Step 0: Symbol Sanitization & BSE Normalizer
    Resolves BSE numeric codes (544563 -> ZELIO) and strips SME / market segment suffixes.
    """
    if bse_reverse_map is None:
        bse_reverse_map = load_bse_reverse_map()

    raw_str = str(ticker).strip()
    resolved = bse_reverse_map.get(raw_str, raw_str)
    
    # Clean ticker for lookup (strip -SM, -ST, -BE, .NS, .BO)
    clean_sym = (
        resolved.replace(".NS", "")
        .replace(".BO", "")
        .replace("-SM", "")
        .replace("-ST", "")
        .replace("-BE", "")
        .strip()
    )
    return resolved, clean_sym


def get_deterministic_stock_industry(ticker, bse_reverse_map=None):
    """
    Executes the 3-Tier Classification Pipeline:
      Tier 1: Canonical Database Resolution
      Tier 2: Exchange / yfinance API Dynamic Fallback
    Returns (resolved_ticker, clean_symbol, sub_industry)
    """
    resolved, clean_sym = resolve_symbol(ticker, bse_reverse_map)

    # Check known ETFs
    if clean_sym.upper() in KNOWN_ETF_THEMES:
        _, sub_ind, _ = KNOWN_ETF_THEMES[clean_sym.upper()]
        return resolved, clean_sym, sub_ind

    # Tier 1: Canonical Master Database Resolution
    imap = load_canonical_industry_map()
    fcache = load_db_fundamentals_cache()

    candidates = [
        f"{resolved}.NS",
        f"{clean_sym}.NS",
        f"{resolved}.BO",
        f"{clean_sym}.BO",
        resolved,
        clean_sym
    ]

    for cand in candidates:
        if cand in imap and imap[cand] and imap[cand] not in ("Other / ETF", "Unknown"):
            return resolved, clean_sym, imap[cand]
        if cand in fcache and fcache[cand] and fcache[cand] not in ("Other / ETF", "Unknown"):
            return resolved, clean_sym, fcache[cand]

    # Tier 2: Dynamic Exchange Fallback (yfinance)
    try:
        import yfinance as yf
        yf_ticker = f"{clean_sym}.NS"
        stock = yf.Ticker(yf_ticker)
        info = stock.info
        fetched_ind = info.get("industry")
        if not fetched_ind:
            fetched_ind = info.get("sector")
            
        if fetched_ind and fetched_ind not in ("Other / ETF", "None", ""):
            persist_industry_to_cache(f"{clean_sym}.NS", fetched_ind)
            return resolved, clean_sym, fetched_ind
    except Exception:
        pass

    # If SME with -SM or -ST suffix, label transparently as SME
    if "-SM" in resolved or "-ST" in resolved or raw_is_sme(resolved):
        return resolved, clean_sym, "SME / Emerging Growth"

    return resolved, clean_sym, "Unclassified / Micro-Cap"


def raw_is_sme(symbol):
    """Check if symbol has typical Indian SME or Series flags."""
    s = str(symbol).upper()
    return s.endswith("-SM") or s.endswith("-ST") or s.endswith("-BE") or (s.isdigit() and len(s) == 6)


def map_sub_industry_to_theme(sub_industry):
    """
    Tier 3: Maps official sub-industry strings to Canonical Thematic Super-Clusters.
    Zero manual ticker mapping.
    """
    if not sub_industry:
        return "📌 Unclassified / Micro-Cap"

    # Exact match in immutable dictionary
    if sub_industry in SUB_INDUSTRY_TO_THEME:
        return SUB_INDUSTRY_TO_THEME[sub_industry]

    # Substring rule match for variations
    sub_lower = sub_industry.lower()
    if "drug" in sub_lower or "pharma" in sub_lower or "medical" in sub_lower or "biotech" in sub_lower or "diagnost" in sub_lower or "health" in sub_lower:
        return "🏥 Pharmaceuticals & Healthcare"
    if "auto" in sub_lower or "motor" in sub_lower or "vehicle" in sub_lower:
        return "🏭 Automotive & Mobility"
    if "chemical" in sub_lower or "agrochem" in sub_lower or "fertiliz" in sub_lower:
        return "🧪 Specialty Chemicals & Materials"
    if "luxury" in sub_lower or "jewel" in sub_lower or "apparel" in sub_lower or "fashion" in sub_lower or "textile" in sub_lower:
        return "💎 Luxury Retail & Jewelry"
    if "cable" in sub_lower or "wire" in sub_lower or "electrical" in sub_lower:
        return "🔌 Electrical Equipment & Cables"
    if "electronic" in sub_lower or "hardware" in sub_lower or "semiconductor" in sub_lower:
        return "🔌 Electronics Hardware & Components"
    if "defense" in sub_lower or "aerospace" in sub_lower or "military" in sub_lower:
        return "🛡️ Defense & Aerospace"
    if "solar" in sub_lower or "wind" in sub_lower or "renewab" in sub_lower:
        return "☀️ Solar & Renewable Energy"
    if "power" in sub_lower or "electric util" in sub_lower or "transmission" in sub_lower or "grid" in sub_lower:
        return "⚡ Power T&D & Utilities"
    if "machinery" in sub_lower or "fabrication" in sub_lower or "engineering equipment" in sub_lower:
        return "⚙️ Industrial Machinery & Engineering"
    if "construction" in sub_lower or "infrastructure" in sub_lower or "civil engineering" in sub_lower:
        return "🏗️ Engineering & Infrastructure"
    if "building" in sub_lower or "cement" in sub_lower or "ceramic" in sub_lower or "glass" in sub_lower:
        return "🏗️ Building Products & Materials"
    if "logistics" in sub_lower or "freight" in sub_lower or "shipping" in sub_lower or "port" in sub_lower:
        return "📦 Logistics & Supply Chain"
    if "steel" in sub_lower or "copper" in sub_lower or "metal" in sub_lower or "mining" in sub_lower:
        return "⛏️ Metals & Mining"
    if "telecom" in sub_lower or "communication" in sub_lower:
        return "📡 Telecom & Communications"
    if "software" in sub_lower or "information tech" in sub_lower or "cloud" in sub_lower:
        return "💻 Software & Technology"
    if "capital market" in sub_lower or "wealth" in sub_lower or "asset manage" in sub_lower or "exchange" in sub_lower:
        return "📈 Capital Markets & Platforms"
    if "bank" in sub_lower or "nbfc" in sub_lower or "credit" in sub_lower or "lending" in sub_lower:
        return "🏦 Banking & Financial Services"
    if "hotel" in sub_lower or "resort" in sub_lower or "hospitality" in sub_lower or "restaurant" in sub_lower:
        return "🏨 Hotels & Premium Hospitality"
    if "food" in sub_lower or "beverage" in sub_lower or "consumer staple" in sub_lower or "fmcg" in sub_lower:
        return "🌾 FMCG & Consumer Staples"
    if "gold etf" in sub_lower or "commodity" in sub_lower or "liquid" in sub_lower:
        return "💰 Cash & Commodities"
    if "sme" in sub_lower:
        return "📌 SME & Emerging Growth"

    return f"📌 {sub_industry}"


def analyze_portfolio_theme_contagion(decisions, bse_reverse_map=None):
    """
    Core CANSLIM Contagion Analysis Pipeline:
      1. Resolves every holding deterministically to its canonical sub-industry & thematic cluster.
      2. Measures thematic capital allocation weight % and rupees invested.
      3. Calculates the Thematic Contagion Index (TCI): % of constituents undergoing institutional distribution.
      4. Evaluates cluster 50D Up/Down Volume Flow.
      5. Generates actionable O'Neil 50% Rule fund mandates.
    """
    if bse_reverse_map is None:
        bse_reverse_map = load_bse_reverse_map()

    theme_buckets = {}
    stock_theme_map = {}

    total_portfolio_weight = 0.0

    for d in decisions:
        raw_ticker = str(d.get("ticker", "")).strip()
        if not raw_ticker or raw_ticker == "BENCHMARK":
            continue

        resolved_sym, clean_sym, sub_industry = get_deterministic_stock_industry(raw_ticker, bse_reverse_map)
        theme = map_sub_industry_to_theme(sub_industry)

        # Portfolio Weight
        loss_m = d.get("loss_metrics", {})
        weight = float(loss_m.get("portfolio_weight", 0.0))
        if weight <= 0.0:
            weight = float(d.get("weight", 0.0))
        total_portfolio_weight += weight

        # Distribution Metrics
        dist_m = d.get("distribution_metrics", {})
        ids_score = int(dist_m.get("ids_score", 0))
        idvd_15d = int(dist_m.get("idvd_count_15d", 0))
        total_dd_15d = int(dist_m.get("total_dist_days_15d", 0))
        cluster_detected = bool(dist_m.get("cluster_detected", False))
        haber_21 = bool(dist_m.get("haber_21_breach", False))
        oneil_50 = bool(dist_m.get("oneil_50_breach", False))
        ud_ratio = float(dist_m.get("ud_volume_ratio_50d", 1.0))
        closes_below_21 = int(dist_m.get("consecutive_closes_below_21", 0))

        # CANSLIM Distributive Stock Flag
        # A stock is distributive if IDS >= 5, or total DDs >= 2, or IDVD >= 2, or broken 21 EMA / 50 SMA
        is_distributive = (
            ids_score >= 5
            or total_dd_15d >= 2
            or idvd_15d >= 2
            or cluster_detected
            or oneil_50
            or (haber_21 and closes_below_21 >= 3)
        )

        stock_record = {
            "raw_ticker": raw_ticker,
            "resolved_symbol": resolved_sym,
            "clean_symbol": clean_sym,
            "sub_industry": sub_industry,
            "theme": theme,
            "weight": weight,
            "ids_score": ids_score,
            "idvd_15d": idvd_15d,
            "total_dd_15d": total_dd_15d,
            "cluster_detected": cluster_detected,
            "haber_21": haber_21,
            "oneil_50": oneil_50,
            "ud_ratio": ud_ratio,
            "is_distributive": is_distributive,
            "status": dist_m.get("status", "🟢 Clean"),
            "prescription": dist_m.get("prescription", "Normal accumulation.")
        }

        stock_theme_map[raw_ticker] = stock_record

        if theme not in theme_buckets:
            theme_buckets[theme] = {
                "theme": theme,
                "sub_industries": set(),
                "constituents": [],
                "total_weight": 0.0,
                "distributive_count": 0,
                "distributive_symbols": [],
                "clean_symbols": [],
                "ud_ratios": []
            }

        bucket = theme_buckets[theme]
        bucket["sub_industries"].add(sub_industry)
        bucket["constituents"].append(stock_record)
        bucket["total_weight"] += weight
        bucket["ud_ratios"].append(ud_ratio)

        if is_distributive:
            bucket["distributive_count"] += 1
            bucket["distributive_symbols"].append(resolved_sym)
        else:
            bucket["clean_symbols"].append(resolved_sym)

    # Compile Thematic Contagion Matrix
    theme_matrix = []
    severe_contagion_themes = []
    elevated_risk_themes = []

    for theme, b in theme_buckets.items():
        n = len(b["constituents"])
        dist_count = b["distributive_count"]
        contagion_rate = (dist_count / n * 100.0) if n > 0 else 0.0
        avg_ud = float(np.mean(b["ud_ratios"])) if b["ud_ratios"] else 1.0
        tot_wt = b["total_weight"]

        # Risk Classification
        is_sme_cluster = "SME" in theme or "Micro-Cap" in theme
        if not is_sme_cluster and n >= 2 and contagion_rate >= 50.0:
            status_badge = "🔥 Severe Contagion"
            status_color = "red"
            mandate = (
                f"🚨 Sector-wide fund liquidation ({dist_count}/{n} stocks distributive). "
                f"Halt buys in {theme}; trim group capital to <10%; exit leaders violating key MAs."
            )
            severe_contagion_themes.append(theme)
        elif not is_sme_cluster and (contagion_rate >= 33.0 or tot_wt >= 25.0):
            status_badge = "⚠️ Elevated Contagion"
            status_color = "yellow"
            mandate = (
                f"⚠️ Supply pressure building ({dist_count}/{n} distributive, {tot_wt:.1f}% exposure). "
                f"Freeze additions; tighten stops to breakeven on lagging positions."
            )
            elevated_risk_themes.append(theme)
        elif is_sme_cluster:
            status_badge = "⚪ Micro-Cap Float"
            status_color = "slate"
            mandate = "Manage via individual ATR/chandelier stops; low systemic basket contagion impact."
        elif contagion_rate == 0.0 and avg_ud >= 1.15:
            status_badge = "🟢 Pristine Accumulation"
            status_color = "green"
            mandate = "Elite institutional sponsorship across cluster. Let winners run; safe to pyramid on proper bases."
        else:
            status_badge = "🟢 Constructive Action"
            status_color = "green"
            mandate = "Normal institutional action. Accumulation intact; enforce normal stop rules."

        theme_matrix.append({
            "theme": theme,
            "capital_weight": tot_wt,
            "constituent_count": n,
            "distributive_count": dist_count,
            "contagion_rate": contagion_rate,
            "avg_ud_ratio": avg_ud,
            "status_badge": status_badge,
            "status_color": status_color,
            "tactical_mandate": mandate,
            "constituents": b["constituents"],
            "distributive_symbols": b["distributive_symbols"],
            "clean_symbols": b["clean_symbols"],
            "sub_industries": sorted(list(b["sub_industries"]))
        })

    # Sort matrix: high contagion and high capital weight at the top
    theme_matrix.sort(key=lambda x: (x["status_color"] == "red", x["contagion_rate"], x["capital_weight"]), reverse=True)

    summary = {
        "total_themes": len(theme_matrix),
        "severe_contagion_count": len(severe_contagion_themes),
        "elevated_risk_count": len(elevated_risk_themes),
        "severe_contagion_themes": severe_contagion_themes,
        "elevated_risk_themes": elevated_risk_themes,
        "has_critical_alert": len(severe_contagion_themes) > 0,
        "total_portfolio_weight": total_portfolio_weight
    }

    return {
        "summary": summary,
        "theme_matrix": theme_matrix,
        "stock_theme_map": stock_theme_map
    }
