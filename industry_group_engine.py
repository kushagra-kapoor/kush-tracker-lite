"""
CANSLIM Industry Group Momentum Matrix Engine
=============================================
Calculates multi-timeframe relative strength rankings, rank velocity deltas,
pack hunting breadth, and actionable CANSLIM execution badges for 100+ granular
sub-industries and curated Indian Alpha thematic clusters.
"""

from __future__ import annotations

import os
import re
import json
import pickle
import urllib.request
import numpy as np
import pandas as pd
import streamlit as st
from risk_metrics import calculate_sortino_ratio, calculate_smooth_rs_atr

MATRIX_PKL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "historical_prices_matrix.pkl")
MAP_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "industry_map.json")


def _get_db_cache_sig() -> tuple:
    """Return DB file mtime/size when running against a local SQLite file, else (0, 0) on Turso Cloud."""
    try:
        from database import DATABASE_PATH
        if DATABASE_PATH and os.path.exists(DATABASE_PATH):
            return _get_file_signature(DATABASE_PATH)
    except Exception:
        pass
    return (0, 0)

# Persistent Snapshot Cache Paths
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
CACHE_INTEL_INDIA_PATH = os.path.join(CACHE_DIR, "industry_group_intelligence_india.json")
CACHE_INTEL_US_PATH = os.path.join(CACHE_DIR, "industry_group_intelligence_us.json")
CACHE_MATRIX_INDIA_PATH = os.path.join(CACHE_DIR, "industry_group_matrix_india.json")
CACHE_MATRIX_US_PATH = os.path.join(CACHE_DIR, "industry_group_matrix_us.json")
CACHE_NICHE_INDIA_PATH = os.path.join(CACHE_DIR, "industry_group_niche_india.json")
CACHE_NICHE_US_PATH = os.path.join(CACHE_DIR, "industry_group_niche_us.json")

CACHE_SCHEMA_VERSION = 3
MIN_RANKED_GROUP_SIZE_CANONICAL = 4
MIN_RANKED_GROUP_SIZE_THEMATIC = 3
EXCLUDED_CANONICAL_GROUPS = {"Shell Companies", "General / Unclassified"}

# =============================================================================
# CANSLIM CANONICAL 145 TAXONOMY & INSTITUTIONAL SYNONYM NORMALIZER
# =============================================================================
CANSLIM_CANONICAL_INDUSTRIES = {
    'Advertising Agencies', 'Aerospace & Defense', 'Agricultural Inputs', 'Airlines', 'Airports & Air Services',
    'Aluminum', 'Apparel Manufacturing', 'Apparel Retail', 'Asset Management', 'Auto & Truck Dealerships',
    'Auto Manufacturers', 'Auto Parts', 'Banks - Diversified', 'Banks - Regional', 'Beverages - Brewers',
    'Beverages - Non-Alcoholic', 'Beverages - Wineries & Distilleries', 'Biotechnology', 'Broadcasting', 'Building Materials',
    'Building Products & Equipment', 'Business Equipment & Supplies', 'Capital Markets', 'Chemicals', 'Coking Coal',
    'Communication Equipment', 'Computer Hardware', 'Confectioners', 'Conglomerates', 'Consulting Services',
    'Consumer Electronics', 'Copper', 'Credit Services', 'Department Stores', 'Diagnostics & Research',
    'Discount Stores', 'Drug Manufacturers - General', 'Drug Manufacturers - Specialty & Generic', 'Education & Training Services', 'Electrical Equipment & Parts',
    'Electronic Components', 'Electronic Gaming & Multimedia', 'Electronics & Computer Distribution', 'Engineering & Construction', 'Entertainment',
    'Farm & Heavy Construction Machinery', 'Farm Products', 'Financial Conglomerates', 'Financial Data & Stock Exchanges', 'Food Distribution',
    'Footwear & Accessories', 'Furnishings, Fixtures & Appliances', 'Gambling', 'Gold', 'Grocery Stores',
    'Health Information Services', 'Healthcare Plans', 'Home Improvement Retail', 'Household & Personal Products', 'Industrial Distribution',
    'Information Technology Services', 'Infrastructure Operations', 'Insurance - Diversified', 'Insurance - Life', 'Insurance - Property & Casualty',
    'Insurance - Reinsurance', 'Insurance - Specialty', 'Insurance Brokers', 'Integrated Freight & Logistics', 'Internet Content & Information',
    'Internet Retail', 'Leisure', 'Lodging', 'Lumber & Wood Production', 'Luxury Goods',
    'Marine Shipping', 'Medical Care Facilities', 'Medical Devices', 'Medical Distribution', 'Medical Instruments & Supplies',
    'Metal Fabrication', 'Mortgage Finance', 'Oil & Gas Drilling', 'Oil & Gas E&P', 'Oil & Gas Equipment & Services',
    'Oil & Gas Integrated', 'Oil & Gas Midstream', 'Oil & Gas Refining & Marketing', 'Other Industrial Metals & Mining', 'Other Precious Metals & Mining',
    'Packaged Foods', 'Packaging & Containers', 'Paper & Paper Products', 'Personal Services', 'Pharmaceutical Retailers',
    'Pollution & Treatment Controls', 'Publishing', 'REIT - Diversified', 'REIT - Healthcare Facilities', 'REIT - Hotel & Motel',
    'REIT - Industrial', 'REIT - Mortgage', 'REIT - Office', 'REIT - Residential', 'REIT - Retail',
    'REIT - Specialty', 'Railroads', 'Real Estate - Development', 'Real Estate - Diversified', 'Real Estate Services',
    'Recreational Vehicles', 'Rental & Leasing Services', 'Residential Construction', 'Resorts & Casinos', 'Restaurants',
    'Scientific & Technical Instruments', 'Security & Protection Services', 'Semiconductor Equipment & Materials', 'Semiconductors', 'Shell Companies',
    'Silver', 'Software - Application', 'Software - Infrastructure', 'Solar', 'Specialty Business Services',
    'Specialty Chemicals', 'Specialty Industrial Machinery', 'Specialty Retail', 'Staffing & Employment Services', 'Steel',
    'Telecom Services', 'Textile Manufacturing', 'Thermal Coal', 'Tobacco', 'Tools & Accessories',
    'Travel Services', 'Trucking', 'Uranium', 'Utilities - Diversified', 'Utilities - Independent Power Producers',
    'Utilities - Regulated Electric', 'Utilities - Regulated Gas', 'Utilities - Regulated Water', 'Utilities - Renewable', 'Waste Management'
}

TAXONOMY_SYNONYM_MAP = {
    'Chemicals: Agricultural': 'Agricultural Inputs',
    'Real Estate Development': 'Real Estate - Development',
    'Investment Banks/Brokers': 'Capital Markets',
    'Wholesale Distributors': 'Industrial Distribution',
    'Packaged Software': 'Software - Application',
    'Pharmaceuticals: Major': 'Drug Manufacturers - General',
    'Pharmaceuticals: Other': 'Drug Manufacturers - Specialty & Generic',
    'Apparel/Footwear Retail': 'Apparel Retail',
    'Apparel/Footwear': 'Apparel Manufacturing',
    'Textiles': 'Textile Manufacturing',
    'Auto Parts: OEM': 'Auto Parts',
    'Automotive Aftermarket': 'Auto Parts',
    'Motor Vehicles': 'Auto Manufacturers',
    'Chemicals: Specialty': 'Specialty Chemicals',
    'Chemicals: Major Diversified': 'Chemicals',
    'Movies/Entertainment': 'Entertainment',
    'Broadcasting/Cable TV': 'Broadcasting',
    'Cable/Satellite TV': 'Broadcasting',
    'Industrial Conglomerates': 'Conglomerates',
    'Electric Utilities': 'Utilities - Regulated Electric',
    'Alternative Power Generation': 'Utilities - Renewable',
    'Gas Distributors': 'Utilities - Regulated Gas',
    'Water Utilities': 'Utilities - Regulated Water',
    'Oil & Gas Production': 'Oil & Gas E&P',
    'Oil Refining/Marketing': 'Oil & Gas Refining & Marketing',
    'Oilfield Services/Equipment': 'Oil & Gas Equipment & Services',
    'Pipelines': 'Oil & Gas Midstream',
    'Coal': 'Thermal Coal',
    'Steel': 'Steel',
    'Aluminum': 'Aluminum',
    'Precious Metals': 'Other Precious Metals & Mining',
    'Industrial Specialties': 'Specialty Industrial Machinery',
    'Electronics/Appliances': 'Consumer Electronics',
    'Electronic Components': 'Electronic Components',
    'Electrical Products': 'Electrical Equipment & Parts',
    'Telecommunications Equipment': 'Communication Equipment',
    'Other Consumer Services': 'Personal Services',
    'Commercial Printing/Forms': 'Publishing',
    'Food: Major Diversified': 'Packaged Foods',
    'Food: Specialty/Candy': 'Confectioners',
    'Food: Meat/Fish/Dairy': 'Packaged Foods',
    'Beverages: Alcoholic': 'Beverages - Wineries & Distilleries',
    'Beverages: Non-Alcoholic': 'Beverages - Non-Alcoholic',
    'Forest Products': 'Paper & Paper Products',
    'Paper': 'Paper & Paper Products',
    'Hotels/Resorts/Cruiselines': 'Lodging',
    'Restaurants': 'Restaurants',
    'Department Stores': 'Department Stores',
    'Discount Stores': 'Discount Stores',
    'Specialty Stores': 'Specialty Retail',
    'Hospital/Nursing Management': 'Medical Care Facilities',
    'Medical/Nursing Services': 'Medical Care Facilities',
    'Medical Specialties': 'Medical Instruments & Supplies',
    'Internet Retail': 'Internet Retail',
    'Trucking': 'Trucking',
    'Air Freight/Couriers': 'Integrated Freight & Logistics',
    'Airlines': 'Airlines',
    'Railroads': 'Railroads',
    'Casinos/Gaming': 'Gambling',
    'Advertising/Marketing Services': 'Advertising Agencies',
    'Data Processing Services': 'Information Technology Services',
    'Computer Processing Hardware': 'Computer Hardware',
    'Computer Communications': 'Communication Equipment',
    'Semiconductors': 'Semiconductors',
    'Semiconductor Equipment': 'Semiconductor Equipment & Materials',
    'Real Estate Investment Trusts': 'REIT - Diversified',
    'Finance/Rental/Leasing': 'Rental & Leasing Services',
    'Major Banks': 'Banks - Diversified',
    'Regional Banks': 'Banks - Regional',
    'Life/Health Insurance': 'Insurance - Life',
    'Property/Casualty Insurance': 'Insurance - Property & Casualty',
    'Insurance Brokers/Services': 'Insurance Brokers',
    'Multi-Line Insurance': 'Insurance - Diversified',
    'Financial Publishing/Services': 'Financial Data & Stock Exchanges',
    'Agricultural Commodities/Milling': 'Farm Products',
    'Metal Fabrication': 'Metal Fabrication',
    'Building Products': 'Building Products & Equipment',
    'Construction Materials': 'Building Materials',
    'Engineering & Construction': 'Engineering & Construction',
    'Miscellaneous Commercial Services': 'Specialty Business Services',
    'Environmental Services': 'Waste Management',
    'Other Transportation': 'Infrastructure Operations',
    'Marine Shipping': 'Marine Shipping',
    'Biotechnology': 'Biotechnology',
    'Financial Conglomerates': 'Financial Conglomerates',
    # Granular GICS Sub-Industries & Non-Canonical Synonyms
    'Agricultural & Farm Machinery': 'Farm & Heavy Construction Machinery',
    'Agricultural Products & Services': 'Farm Products',
    'Air Freight & Logistics': 'Integrated Freight & Logistics',
    'Apparel, Accessories & Luxury Goods': 'Apparel Manufacturing',
    'Application Software': 'Software - Application',
    'Asset Management & Custody Banks': 'Asset Management',
    'Automobile Manufacturers': 'Auto Manufacturers',
    'Automotive Parts & Equipment': 'Auto Parts',
    'Automotive Retail': 'Auto & Truck Dealerships',
    'Broadline Retail': 'Department Stores',
    'Cable & Satellite': 'Broadcasting',
    'Cargo Ground Transportation': 'Trucking',
    'Casinos & Gaming': 'Gambling',
    'Commodity Chemicals': 'Chemicals',
    'Computer & Electronics Retail': 'Specialty Retail',
    'Computer Peripherals': 'Computer Hardware',
    'Construction & Engineering': 'Engineering & Construction',
    'Construction Machinery & Heavy Transportation Equipment': 'Farm & Heavy Construction Machinery',
    'Consumer Finance': 'Credit Services',
    'Consumer Staples Merchandise Retail': 'Grocery Stores',
    'Containers/Packaging': 'Packaging & Containers',
    'Data Center REITs': 'REIT - Specialty',
    'Data Processing & Outsourced Services': 'Information Technology Services',
    'Distillers & Vintners': 'Beverages - Wineries & Distilleries',
    'Diversified Banks': 'Banks - Diversified',
    'Diversified Support Services': 'Specialty Business Services',
    'Electrical Components & Equipment': 'Electrical Equipment & Parts',
    'Electronic Equipment & Instruments': 'Scientific & Technical Instruments',
    'Electronic Equipment/Instruments': 'Scientific & Technical Instruments',
    'Electronic Manufacturing Services': 'Electronic Components',
    'Electronic Production Equipment': 'Semiconductor Equipment & Materials',
    'Electronics Distributors': 'Electronics & Computer Distribution',
    'Electronics/Appliance Stores': 'Specialty Retail',
    'Environmental & Facilities Services': 'Waste Management',
    'Fertilizers & Agricultural Chemicals': 'Agricultural Inputs',
    'Financial Exchanges & Data': 'Financial Data & Stock Exchanges',
    'Food Distributors': 'Food Distribution',
    'Food Retail': 'Grocery Stores',
    'Gas Utilities': 'Utilities - Regulated Gas',
    'Health Care Distributors': 'Medical Distribution',
    'Health Care Equipment': 'Medical Devices',
    'Health Care Facilities': 'Medical Care Facilities',
    'Health Care REITs': 'REIT - Healthcare Facilities',
    'Health Care Services': 'Health Information Services',
    'Health Care Supplies': 'Medical Instruments & Supplies',
    'Health Care Technology': 'Health Information Services',
    'Heavy Electrical Equipment': 'Electrical Equipment & Parts',
    'Homebuilding': 'Residential Construction',
    'Home Furnishings': 'Furnishings, Fixtures & Appliances',
    'Homefurnishing Retail': 'Home Improvement Retail',
    'Hotel & Resort REITs': 'REIT - Hotel & Motel',
    'Hotels, Resorts & Cruise Lines': 'Lodging',
    'Hotels/Resorts/Cruise Lines': 'Lodging',
    'Household Products': 'Household & Personal Products',
    'Human Resource & Employment Services': 'Staffing & Employment Services',
    'IT Consulting & Other Services': 'Information Technology Services',
    'Independent Power Producers & Energy Traders': 'Utilities - Independent Power Producers',
    'Industrial Gases': 'Chemicals',
    'Industrial Machinery': 'Specialty Industrial Machinery',
    'Industrial Machinery & Supplies & Components': 'Specialty Industrial Machinery',
    'Industrial REITs': 'REIT - Industrial',
    'Integrated Oil': 'Oil & Gas Integrated',
    'Integrated Oil & Gas': 'Oil & Gas Integrated',
    'Integrated Telecommunication Services': 'Telecom Services',
    'Interactive Home Entertainment': 'Electronic Gaming & Multimedia',
    'Interactive Media & Services': 'Internet Content & Information',
    'Internet Services & Infrastructure': 'Software - Infrastructure',
    'Internet Software/Services': 'Software - Infrastructure',
    'Investment Banking & Brokerage': 'Capital Markets',
    'Investment Managers': 'Asset Management',
    'Investment Trusts/Mutual Funds': 'Asset Management',
    'Leisure Products': 'Leisure',
    'Life & Health Insurance': 'Insurance - Life',
    'Life Sciences Tools & Services': 'Diagnostics & Research',
    'Managed Health Care': 'Healthcare Plans',
    'Medical Distributors': 'Medical Distribution',
    'Metal, Glass & Plastic Containers': 'Packaging & Containers',
    'Movies & Entertainment': 'Entertainment',
    'Multi-Family Residential REITs': 'REIT - Residential',
    'Multi-Sector Holdings': 'Conglomerates',
    'Multi-Utilities': 'Utilities - Diversified',
    'Office REITs': 'REIT - Office',
    'Oil & Gas Exploration & Production': 'Oil & Gas E&P',
    'Oil & Gas Storage & Transportation': 'Oil & Gas Midstream',
    'Other Specialized REITs': 'REIT - Specialty',
    'Other Specialty Retail': 'Specialty Retail',
    'Other Consumer Specialties': 'Personal Services',
    'Other Metals/Minerals': 'Other Industrial Metals & Mining',
    'Packaged Foods & Meats': 'Packaged Foods',
    'Passenger Ground Transportation': 'Travel Services',
    'Personal Care Products': 'Household & Personal Products',
    'Personnel Services': 'Staffing & Employment Services',
    'Property & Casualty Insurance': 'Insurance - Property & Casualty',
    'Rail Transportation': 'Railroads',
    'Reinsurance': 'Insurance - Reinsurance',
    'Research & Consulting Services': 'Consulting Services',
    'Retail REITs': 'REIT - Retail',
    'Self-Storage REITs': 'REIT - Industrial',
    'Semiconductor Materials & Equipment': 'Semiconductor Equipment & Materials',
    'Single-Family Residential REITs': 'REIT - Residential',
    'Soft Drinks & Non-alcoholic Beverages': 'Beverages - Non-Alcoholic',
    'Specialized Consumer Services': 'Personal Services',
    'Specialty Telecommunications': 'Telecom Services',
    'Systems Software': 'Software - Infrastructure',
    'Technology Distributors': 'Electronics & Computer Distribution',
    'Technology Hardware, Storage & Peripherals': 'Computer Hardware',
    'Telecom Tower REITs': 'REIT - Specialty',
    'Timber REITs': 'REIT - Specialty',
    'Trading Companies & Distributors': 'Industrial Distribution',
    'Transaction & Payment Processing Services': 'Capital Markets',
    'Trucks/Construction/Farm Machinery': 'Farm & Heavy Construction Machinery',
    'Wireless Telecommunication Services': 'Telecom Services',
    'Wireless Telecommunications': 'Telecom Services',
    # TradingView / Live IN & US Non-Canonical Taxonomy Mappings
    'Catalog/Specialty Distribution': 'Specialty Retail',
    'Consumer Sundries': 'Household & Personal Products',
    'Contract Drilling': 'Oil & Gas Drilling',
    'Drugstore Chains': 'Pharmaceutical Retailers',
    'Home Improvement Chains': 'Home Improvement Retail',
    'Household/Personal Care': 'Household & Personal Products',
    'Major Telecommunications': 'Telecom Services',
    'Media Conglomerates': 'Entertainment',
    'Miscellaneous Manufacturing': 'Specialty Industrial Machinery',
    'Office Equipment/Supplies': 'Business Equipment & Supplies',
    'Pharmaceuticals: Generic': 'Drug Manufacturers - Specialty & Generic',
    'Publishing: Books/Magazines': 'Publishing',
    'Recreational Products': 'Leisure',
    'Savings Banks': 'Banks - Regional',
    'Services To The Health Industry': 'Health Information Services',
    'Specialty Insurance': 'Insurance - Specialty',
    'Tools & Hardware': 'Tools & Accessories',
    'General / Unclassified': 'Shell Companies',
}

_CANONICAL_LOWER_MAP = {c.lower(): c for c in CANSLIM_CANONICAL_INDUSTRIES}
_SYNONYM_LOWER_MAP = {k.lower(): v for k, v in TAXONOMY_SYNONYM_MAP.items() if v in CANSLIM_CANONICAL_INDUSTRIES}
_LOGGED_UNKNOWN_TAXONOMY = set()


def _lookup_canonical_or_synonym(raw_text: str):
    """Exact case-insensitive lookup against canonical 145 groups and curated synonym table."""
    if not raw_text:
        return None
    cleaned = str(raw_text).strip()
    if not cleaned or cleaned.lower() in {"unknown", "none", "nan", "—", "-", "n/a", "general"}:
        return None
    if cleaned in CANSLIM_CANONICAL_INDUSTRIES:
        return cleaned
    lower = cleaned.lower()
    if lower in _CANONICAL_LOWER_MAP:
        return _CANONICAL_LOWER_MAP[lower]
    if lower in _SYNONYM_LOWER_MAP:
        return _SYNONYM_LOWER_MAP[lower]
    return None


def normalize_taxonomy(raw_industry: str, raw_sector: str = "") -> str:
    """
    Normalizes TradingView / Yahoo Finance taxonomy into canonical 145 CANSLIM industry groups
    using exact case-insensitive canonical & synonym resolution (zero substring corruption).
    """
    ind_str = "" if raw_industry is None else str(raw_industry).strip()
    sec_str = "" if raw_sector is None else str(raw_sector).strip()

    if ind_str.lower() in {"", "unknown", "none", "nan", "general", "general / unclassified"}:
        resolved = _lookup_canonical_or_synonym(sec_str)
        return resolved if resolved else "General / Unclassified"

    resolved = _lookup_canonical_or_synonym(ind_str)
    if resolved == "Shell Companies" and ind_str.lower() == "general / unclassified":
        return "General / Unclassified"
    if resolved:
        return resolved

    sec_resolved = _lookup_canonical_or_synonym(sec_str)
    if sec_resolved:
        return sec_resolved

    if ind_str not in _LOGGED_UNKNOWN_TAXONOMY:
        _LOGGED_UNKNOWN_TAXONOMY.add(ind_str)
        print(f"[IndustryGroupEngine] Unmapped industry taxonomy -> General / Unclassified: {ind_str!r}")
    return "General / Unclassified"

def auto_enrich_missing_industries(universe_scope: str = "India (NSE/BSE)", max_enrich: int = 1000) -> int:
    """
    Dual-Channel Auto-Discovery Engine:
    Resolves missing equities into accredited CANSLIM industry groups.
    Priority 1: TradingView Batch Scanner (sub-second resolution of 500+ stocks).
    Priority 2: Yahoo Finance fallback for any remaining gaps.
    Saves new mappings to industry_map.json and SQLite fundamentals_cache.
    """
    if not os.path.exists(MATRIX_PKL_PATH):
        return 0
        
    try:
        with open(MATRIX_PKL_PATH, "rb") as f:
            matrix = pickle.load(f)
    except Exception:
        return 0
        
    if isinstance(matrix.columns, pd.MultiIndex):
        close_df = matrix.xs("Close", level=1, axis=1)
    else:
        close_df = matrix
        
    # Load current active mapping union
    ind_map = {}
    if os.path.exists(MAP_JSON_PATH):
        try:
            with open(MAP_JSON_PATH, "r", encoding="utf-8") as f:
                ind_map = json.load(f)
        except Exception:
            ind_map = {}
            
    try:
        from database import read_sql_df
        df_fc = read_sql_df(
            "SELECT ticker, industry FROM fundamentals_cache WHERE industry IS NOT NULL AND industry != '' AND industry != 'Unknown'"
        )
        if df_fc is not None and not df_fc.empty:
            for d_t, d_ind in zip(df_fc["ticker"].tolist(), df_fc["industry"].tolist()):
                if d_t and d_ind and d_t not in ind_map:
                    ind_map[d_t] = d_ind
    except Exception:
        pass

    # Filter targets for scope
    is_india = universe_scope.startswith("India")
    missing_targets = []
    
    for c in close_df.columns:
        if is_india:
            if not (c.endswith(".NS") or c.endswith(".BO")):
                continue
            # Data hygiene: filter dead scrips (< 5 bars)
            if close_df[c].dropna().shape[0] < 5:
                continue
            clean = c.replace(".NS", "").replace(".BO", "")
            # Filter numeric scrips
            if clean.isdigit() and len(clean) >= 5:
                continue
            # Check if mapped
            if c in ind_map or clean in ind_map:
                continue
            missing_targets.append(c)
        else:
            if c.endswith(".NS") or c.endswith(".BO"):
                continue
            if close_df[c].dropna().shape[0] < 5:
                continue
            if c in ind_map:
                continue
            missing_targets.append(c)
            
    if not missing_targets:
        return 0
        
    missing_targets = missing_targets[:max_enrich]
    newly_resolved = {}
    
    # Priority 1: TradingView Batch Scanner
    chunk_size = 250
    scanner_url = "https://scanner.tradingview.com/india/scan" if is_india else "https://scanner.tradingview.com/america/scan"
    
    for i in range(0, len(missing_targets), chunk_size):
        chunk = missing_targets[i:i + chunk_size]
        tv_symbols = []
        sym_to_raw = {}
        for raw in chunk:
            clean = raw.replace(".NS", "").replace(".BO", "")
            clean_query = clean.replace("-SM", "").replace("-ST", "").replace("-BE", "").replace("-BZ", "")
            if is_india:
                prefix = "BSE:" if raw.endswith(".BO") else "NSE:"
            else:
                prefix = ""
            full_tv = prefix + clean_query
            tv_symbols.append(full_tv)
            sym_to_raw[full_tv] = raw
            
        req_body = {
            "symbols": {"tickers": tv_symbols},
            "columns": ["description", "sector", "industry"]
        }
        try:
            req = urllib.request.Request(
                scanner_url,
                data=json.dumps(req_body).encode("utf-8"),
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                for item in res_data.get("data", []):
                    s = item.get("s")
                    d = item.get("d", [])
                    raw_sec = d[1] if len(d) > 1 and d[1] else ""
                    raw_ind = d[2] if len(d) > 2 and d[2] else ""
                    if raw_ind or raw_sec:
                        norm_ind = normalize_taxonomy(raw_ind, raw_sec)
                        raw_sym = sym_to_raw.get(s)
                        if raw_sym:
                            newly_resolved[raw_sym] = norm_ind
                            clean_sym = raw_sym.replace(".NS", "").replace(".BO", "")
                            newly_resolved[clean_sym] = norm_ind
        except Exception as e:
            print(f"[IndustryGroupEngine] TV Scanner error on chunk {i}: {e}")
            
    # Priority 2: Yahoo Finance Fallback for any unresolved
    unresolved_after_tv = []
    for raw in missing_targets:
        clean = raw.replace(".NS", "").replace(".BO", "")
        if raw not in newly_resolved and clean not in newly_resolved:
            unresolved_after_tv.append(raw)
            
    if unresolved_after_tv:
        try:
            import yfinance as yf
            for raw in unresolved_after_tv[:50]:
                try:
                    t_obj = yf.Ticker(raw)
                    yf_ind = t_obj.info.get("industry")
                    yf_sec = t_obj.info.get("sector")
                    if yf_ind or yf_sec:
                        norm_ind = normalize_taxonomy(yf_ind, yf_sec)
                        newly_resolved[raw] = norm_ind
                        clean_sym = raw.replace(".NS", "").replace(".BO", "")
                        newly_resolved[clean_sym] = norm_ind
                except Exception:
                    pass
        except Exception:
            pass
            
    if not newly_resolved:
        return 0
        
    # Persist newly resolved to industry_map.json
    try:
        if os.path.exists(MAP_JSON_PATH):
            with open(MAP_JSON_PATH, "r", encoding="utf-8") as f:
                full_map = json.load(f)
        else:
            full_map = {}
        full_map.update(newly_resolved)
        with open(MAP_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(full_map, f, indent=2)
    except Exception as e:
        print(f"[IndustryGroupEngine] Error saving to industry_map.json: {e}")
        
    # Persist newly resolved to fundamentals_cache via shared database adapter
    try:
        from database import get_connection
        conn = get_connection()
        cur = conn.cursor()
        for sym, ind in newly_resolved.items():
            cur.execute(
                """
                INSERT INTO fundamentals_cache (ticker, industry, updated_at)
                VALUES (?, ?, datetime('now'))
                ON CONFLICT(ticker) DO UPDATE SET
                    industry = excluded.industry,
                    updated_at = excluded.updated_at
                """,
                (sym, ind),
            )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[IndustryGroupEngine] Error updating fundamentals_cache: {e}")
            
    return len(newly_resolved)

# =============================================================================
# CURATED INDIAN ALPHA THEMATIC CLUSTERS
# High-conviction growth themes dominating the current Indian capex & bull cycle
# =============================================================================
INDIAN_ALPHA_THEMES = {
    "⚡ Power T&D & Transformers": [
        "TRIL.NS", "VOLTAMP.NS", "APARINDS.NS", "SCHNEIDER.NS", "BHEL.NS",
        "GEPIL.NS", "POWERINDIA.NS", "KEC.NS", "KALPATPOWR.NS", "TRANSFORME.BO"
    ],
    "🛡️ Defense & Aerospace Systems": [
        "BEL.NS", "HAL.NS", "DATAPATTNS.NS", "ZENTEC.NS", "ASTRAMICRO.NS",
        "PARAS.NS", "SOLARINDS.NS", "BDL.NS", "COCHINSHIP.NS", "MAZDOCK.NS",
        "GRSE.NS", "SIGMAADV.NS", "SWANDEF.NS"
    ],
    "☀️ Solar, Wind & Renewable EPC": [
        "WAAREEENER.NS", "PREMIERENE.NS", "SUZLON.NS", "INOXWIND.NS", "SWENERGY.NS",
        "KPIGREEN.NS", "BORORENEW.NS", "GENSOL.NS", "ADANIGREEN.NS"
    ],
    "🔌 EMS & Electronics Hardware": [
        "DIXON.NS", "KAYNES.NS", "SYRMA.NS", "CYIENTDLM.NS", "AMBER.NS",
        "PGEL.NS", "AVALON.NS", "CENTRUM.NS", "DCXINDIA.NS"
    ],
    "🚆 Railways & Transit Infra": [
        "TITAGARH.NS", "JWL.NS", "RVNL.NS", "IRFC.NS", "TEXRAIL.NS",
        "RITES.NS", "RAILTEL.NS", "CONCOR.NS", "BEML.NS", "IRCTC.NS"
    ],
    "📈 Capital Markets & Wealth Platforms": [
        "BSE.NS", "MCX.NS", "CDSL.NS", "CAMS.NS", "ANGELONE.NS",
        "360ONE.NS", "MOTILALOFS.NS", "NUVAMA.NS", "GEOJITFSL.NS", "ANANDRATHI.NS", "KFINTECH.NS"
    ],
    "🔌 Wires, Cables & EPC": [
        "POLYCAB.NS", "RRKABEL.NS", "KEI.NS", "FINCABLES.NS", "STERTOOLS.NS", "DYNAMIC.NS", "APARINDS.NS"
    ],
    "🏥 Hospitals & Specialty Healthcare": [
        "APOLLOHOSP.NS", "MAXHEALTH.NS", "FORTIS.NS", "RAINBOW.NS", "NH.NS",
        "KIMS.NS", "MEDANTA.NS", "YATHARTH.NS", "ASTERDM.NS"
    ],
    "💎 Retail Luxury, Gold & Jewelry": [
        "TITAN.NS", "KALYANKJIL.NS", "SENCO.NS", "TBZ.NS", "PNGSREVA.NS", "THANGAMAYL.NS", "ETHOSLTD.NS"
    ],
    "🏗️ Real Estate Developers & Infra": [
        "DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "MACROTECH.NS", "PRESTIGE.NS",
        "BRIGADE.NS", "SOBHA.NS", "PURVA.NS"
    ],
    "🧪 Specialty Chemicals & Agro": [
        "PIIND.NS", "DEEPAKNTR.NS", "NAVINFLUOR.NS", "AARTIIND.NS", "TATACHEM.NS",
        "SRF.NS", "FINEORG.NS", "NEOGEN.NS", "COROMANDEL.NS", "SUMICHEM.NS"
    ],
    "🏨 Hotels & Premium Hospitality": [
        "INDHOTEL.NS", "EIHOTEL.NS", "CHALET.NS", "LEMONONTRE.NS", "TAJGVK.NS", "ORIENTHOT.NS"
    ],
    "💻 IT Midcap Growth & Digital": [
        "PERSISTENT.NS", "COFORGE.NS", "KPITTECH.NS", "LTTS.NS", "TATAELXSI.NS",
        "CYIENT.NS", "BSOFT.NS", "SONATSOFTW.NS", "NETWEB.NS"
    ],
    "📦 Logistics, Ports & Cold Chain": [
        "ADANIPORTS.NS", "ALLCARGO.NS", "MAHLOG.NS", "TCIEXP.NS", "BLUEDART.NS",
        "GATEWAY.NS", "VRL.NS", "DELHIVERY.NS", "SCI.NS"
    ],
    "🏭 Auto Ancillaries & Precision Machining": [
        "BHARATFORG.NS", "SONACOMS.NS", "CIEINDIA.NS", "CRAFTSMAN.NS", "SUPRAJIT.NS",
        "RAMKRISNA.NS", "UNOMINDA.NS", "MOTHERSON.NS"
    ],
    "💰 High-Growth NBFCs & Retail Lenders": [
        "BAJFINANCE.NS", "BAJAJFINSV.NS", "CHOLAFIN.NS", "SHRIRAMFIN.NS", "MUTHOOTFIN.NS",
        "MANAPPURAM.NS", "POONAWALLA.NS"
    ],
    "🌾 Sugar, Distilleries & Biofuels": [
        "BALRAMCHIN.NS", "PRAJSIND.NS", "TRIVENI.NS", "EIDPARRY.NS", "DWARKESH.NS", "RENUKA.NS"
    ],
    "🧱 Cement & Building Materials": [
        "ULTRACEMCO.NS", "AMBUJACEM.NS", "SHREECEM.NS", "DALBHARAT.NS", "JKCEMENT.NS",
        "SAGCEM.NS", "RAMCOCEM.NS"
    ],
    "🔋 Battery & EV Ecosystem": [
        "EXIDEIND.NS", "AMARAJABAT.NS", "TATACHEM.NS", "HBLPOWER.NS", "SERVOTECH.NS",
        "OLECTRA.NS", "JBMMA.NS"
    ],
    "🚢 Shipbuilders & Heavy Marine": [
        "COCHINSHIP.NS", "MAZDOCK.NS", "GRSE.NS"
    ],
    "💧 Water Pipes & Infrastructure": [
        "VASTECH.NS", "ASTRAL.NS", "SUPREMEIND.NS", "PRINCEPIPE.NS", "JISLJALEQS.NS",
        "FINPIPE.NS", "SHAKTIPUMP.NS"
    ],
    "🧬 CDMO & Diagnostics": [
        "DIVISLAB.NS", "SUVENPHAR.NS", "LAURUSLABS.NS", "LALPATHLAB.NS", "METROPOLIS.NS",
        "VIJAYA.NS", "SYNGENE.NS"
    ],
    "🏬 Consumer Durables & Appliances": [
        "VOLTAS.NS", "BLUESTARCO.NS", "HAVELLS.NS", "CROMPTON.NS", "WHIRLPOOL.NS", "SYMPHONY.NS"
    ]
}

def _get_file_signature(file_path: str) -> tuple:
    """Return deterministic (st_mtime_ns, st_size) signature for cache invalidation."""
    try:
        st_res = os.stat(file_path)
        return (int(st_res.st_mtime_ns), int(st_res.st_size))
    except OSError:
        return (0, 0)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_price_history_matrix_cached(include_volume: bool, include_low: bool, pkl_sig: tuple):
    """Mtime-keyed cached reader for historical_prices_matrix.pkl."""
    if not pkl_sig or pkl_sig == (0, 0) or not os.path.exists(MATRIX_PKL_PATH):
        empty = pd.DataFrame()
        if include_volume and include_low:
            return empty, empty, empty, empty
        if include_volume or include_low:
            return empty, empty, empty
        return empty, empty

    with open(MATRIX_PKL_PATH, "rb") as f:
        matrix = pickle.load(f)

    if isinstance(matrix.columns, pd.MultiIndex):
        close_df = matrix.xs("Close", level=1, axis=1)
        high_df = matrix.xs("High", level=1, axis=1) if "High" in matrix.columns.levels[1] else close_df
        low_df = matrix.xs("Low", level=1, axis=1) if (include_low and "Low" in matrix.columns.levels[1]) else close_df
        volume_df = matrix.xs("Volume", level=1, axis=1) if (include_volume and "Volume" in matrix.columns.levels[1]) else pd.DataFrame(index=close_df.index, columns=close_df.columns).fillna(0.0)
    else:
        close_df = matrix
        high_df = matrix
        low_df = matrix
        volume_df = pd.DataFrame(index=close_df.index, columns=close_df.columns).fillna(0.0)

    if include_volume and include_low:
        return close_df, high_df, low_df, volume_df
    if include_volume:
        return close_df, high_df, volume_df
    if include_low:
        return close_df, high_df, low_df
    return close_df, high_df


def load_price_history_matrix(include_volume: bool = False, include_low: bool = False):
    """
    Loads historical_prices_matrix.pkl (auto-invalidating on mtime/size changes)
    and returns close_df and high_df (plus optional low_df / volume_df).
    """
    pkl_sig = _get_file_signature(MATRIX_PKL_PATH)
    return _load_price_history_matrix_cached(include_volume, include_low, pkl_sig)


load_price_history_matrix.clear = _load_price_history_matrix_cached.clear


def _read_fundamentals_industry_rows(where_clause: str = "") -> list[tuple]:
    """Read (ticker, industry) rows from fundamentals_cache via shared Turso/SQLite adapter."""
    try:
        from database import read_sql_df
        sql = (
            "SELECT ticker, industry FROM fundamentals_cache "
            "WHERE industry IS NOT NULL AND industry != '' AND industry != 'Unknown'"
        )
        if where_clause:
            sql += f" AND ({where_clause})"
        df_ind = read_sql_df(sql)
        if df_ind is None or df_ind.empty:
            return []
        return list(zip(df_ind["ticker"].tolist(), df_ind["industry"].tolist()))
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def _get_group_constituents_map_cached(
    taxonomy: str,
    universe_scope: str,
    map_sig: tuple,
    db_sig: tuple,
    gics_sig: tuple,
):
    """Mtime-keyed cached constituent builder partitioned cleanly by universe_scope."""
    if taxonomy == "thematic":
        return INDIAN_ALPHA_THEMES

    from price_history_manager import is_excluded_etf_or_index

    scope_u = str(universe_scope or "").strip().upper()
    is_us_scope = scope_u.startswith("US")
    is_in_scope = bool(scope_u) and not is_us_scope

    group_map = {}
    if os.path.exists(MAP_JSON_PATH):
        try:
            with open(MAP_JSON_PATH, "r", encoding="utf-8") as f:
                raw_json = json.load(f)

            if is_in_scope:
                ind_map = {
                    t: ind for t, ind in raw_json.items()
                    if str(t).upper().endswith((".NS", ".BO")) and not is_excluded_etf_or_index(t)
                }
                for d_t, d_ind in _read_fundamentals_industry_rows("ticker LIKE '%.NS' OR ticker LIKE '%.BO'"):
                    if d_t and d_t not in ind_map and d_ind and not is_excluded_etf_or_index(d_t):
                        ind_map[d_t] = d_ind
            elif is_us_scope:
                db_us = {}
                for d_t, d_ind in _read_fundamentals_industry_rows(
                    "ticker NOT LIKE '%.NS' AND ticker NOT LIKE '%.BO' AND ticker NOT LIKE '^%'"
                ):
                    if d_t and d_ind and not is_excluded_etf_or_index(d_t):
                        db_us[d_t] = d_ind
                gics_flat = {}
                gics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sp500_gics_map.json")
                if os.path.exists(gics_path):
                    try:
                        with open(gics_path, "r", encoding="utf-8") as gf:
                            for ut, uinfo in json.load(gf).items():
                                if isinstance(uinfo, dict) and not is_excluded_etf_or_index(ut):
                                    u_ind = uinfo.get("sub_industry") or uinfo.get("industry") or uinfo.get("sector")
                                    if u_ind:
                                        gics_flat[ut] = u_ind
                    except Exception:
                        pass
                us_active = set(db_us.keys()) | set(gics_flat.keys())
                try:
                    from universe_security_master import load_universe_tickers
                    us_active |= {str(t).strip().upper() for t in (load_universe_tickers("US", include_etfs=False) or [])}
                except Exception:
                    try:
                        from us_universe_manager import get_broad_us_universe
                        us_active |= {str(t).strip().upper() for t in (get_broad_us_universe() or [])}
                    except Exception:
                        pass
                in_bases = {str(k).rsplit(".", 1)[0].upper() for k in raw_json if str(k).upper().endswith((".NS", ".BO"))}
                ind_map = {}
                for t, ind in raw_json.items():
                    tu = str(t).strip().upper()
                    if tu.endswith((".NS", ".BO")) or is_excluded_etf_or_index(tu):
                        continue
                    if tu in in_bases and tu not in us_active:
                        continue
                    if ind:
                        ind_map[t] = ind
                for d_t, d_ind in db_us.items():
                    ind_map.setdefault(d_t, d_ind)
                for ut, u_ind in gics_flat.items():
                    ind_map.setdefault(ut, u_ind)
            else:
                ind_map = dict(raw_json)
                for d_t, d_ind in _read_fundamentals_industry_rows():
                    if d_t and d_t not in ind_map and d_ind:
                        ind_map[d_t] = d_ind
                gics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sp500_gics_map.json")
                if os.path.exists(gics_path):
                    try:
                        with open(gics_path, "r", encoding="utf-8") as gf:
                            gics_data = json.load(gf)
                        for ut, uinfo in gics_data.items():
                            if ut not in ind_map and isinstance(uinfo, dict):
                                u_ind = uinfo.get("sub_industry") or uinfo.get("industry") or uinfo.get("sector")
                                if u_ind:
                                    ind_map[ut] = u_ind
                    except Exception:
                        pass

            ns_bases = (
                {str(k).rsplit(".", 1)[0].upper() for k in ind_map if str(k).upper().endswith(".NS")}
                if is_in_scope else set()
            )
            for t, ind in ind_map.items():
                if is_in_scope and str(t).upper().endswith(".BO"):
                    if str(t).rsplit(".", 1)[0].upper() in ns_bases:
                        continue
                if ind and ind != 'Unknown' and not is_excluded_etf_or_index(t):
                    canon_ind = normalize_taxonomy(ind)
                    if canon_ind and canon_ind != 'Unknown':
                        group_map.setdefault(canon_ind, []).append(t)
            return {k: v for k, v in group_map.items() if len(v) >= 1}
        except Exception:
            pass

    where_fc = ""
    if is_in_scope:
        where_fc = "ticker LIKE '%.NS' OR ticker LIKE '%.BO'"
    elif is_us_scope:
        where_fc = "ticker NOT LIKE '%.NS' AND ticker NOT LIKE '%.BO' AND ticker NOT LIKE '^%'"
    fc_rows = _read_fundamentals_industry_rows(where_fc)
    if fc_rows:
        for t, ind in fc_rows:
            if t and ind and not is_excluded_etf_or_index(t):
                canon_ind = normalize_taxonomy(ind)
                if canon_ind and canon_ind != 'Unknown':
                    group_map.setdefault(canon_ind, []).append(t)
        if group_map:
            return group_map

    try:
        from database import get_all_fundamentals_cache
        mkt_arg = "IN" if is_in_scope else ("US" if is_us_scope else None)
        cache = get_all_fundamentals_cache(market=mkt_arg)
        for t, d in cache.items():
            ind = d.get('industry')
            if ind and ind != 'Unknown' and not is_excluded_etf_or_index(t):
                canon_ind = normalize_taxonomy(ind)
                if is_us_scope:
                    sym = t
                else:
                    sym = t if str(t).endswith('.NS') or str(t).endswith('.BO') else f"{t}.NS"
                group_map.setdefault(canon_ind, []).append(sym)
        return {k: v for k, v in group_map.items() if len(v) >= 1}
    except Exception:
        pass

    return group_map


def get_group_constituents_map(taxonomy: str = "canonical", universe_scope: str = None):
    """
    Returns a mapping of group_name -> list of tickers, partitioned strictly by `universe_scope`
    and automatically invalidated on disk changes to industry_map.json, SQLite, or GICS map.
    """
    if taxonomy == "thematic":
        return INDIAN_ALPHA_THEMES
    gics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sp500_gics_map.json")
    return _get_group_constituents_map_cached(
        taxonomy=taxonomy,
        universe_scope=str(universe_scope or ""),
        map_sig=_get_file_signature(MAP_JSON_PATH),
        db_sig=_get_db_cache_sig(),
        gics_sig=_get_file_signature(gics_path),
    )


get_group_constituents_map.clear = _get_group_constituents_map_cached.clear


def _resolve_group_map(taxonomy: str = "canonical", universe_scope: str = None) -> dict:
    try:
        return get_group_constituents_map(taxonomy=taxonomy, universe_scope=universe_scope)
    except TypeError:
        return get_group_constituents_map(taxonomy=taxonomy)


def classify_stock_action_state(
    dist_pivot: float,
    dist_ema: float,
    above_50sma: bool = True,
    above_200sma: bool = True,
) -> tuple[str, str, int]:
    """
    Assigns CANSLIM execution status badge and priority rank:
    - 🟢 IN BUY ZONE (0 to +5.5% from pivot, above 50 SMA & >= -2% 21 EMA): 1
    - 🔵 RETEST (-2.0% to 0% from pivot, above 50 SMA & >= -2% 21 EMA): 2
    - ⏳ COILING (-5.0% to -2.0% from pivot, >= -2.5% 21 EMA): 3
    - 🟠 EXTENDED (>5.5% above pivot): 4
    - ⏳ BASE (-20% to -5% constructive base): 5
    - 🔴 FAILED (< -20% below pivot or broken key moving averages): 6
    """
    if np.isnan(dist_pivot) or np.isnan(dist_ema):
        return "⏳ BASE", "base", 5
    if dist_pivot < -20.0 or (not above_50sma and dist_ema < -2.0) or (not above_200sma and dist_pivot < -8.0):
        return "🔴 FAILED", "failed", 6
    elif dist_pivot > 5.5:
        return "🟠 EXTENDED", "extended", 4
    elif 0.0 <= dist_pivot <= 5.5 and above_50sma and dist_ema >= -2.0:
        return "🟢 IN BUY ZONE", "buy_zone", 1
    elif -2.0 <= dist_pivot < 0.0 and above_50sma and dist_ema >= -2.0:
        return "🔵 RETEST", "retest", 2
    elif -5.0 <= dist_pivot < -2.0 and dist_ema >= -2.5:
        return "⏳ COILING", "coiling", 3
    elif dist_ema < -5.0 and not above_50sma:
        return "🔴 FAILED", "failed", 6
    else:
        return "⏳ BASE", "base", 5

def classify_rotation_status(cur_rank: int, delta_1m: int, total_groups: int = None) -> tuple[str, str]:
    """
    Scale-invariant institutional rotation velocity classifier:
    uses percentile rank thresholds when total_groups is provided while preserving
    exact legacy 2-arg call compatibility.
    """
    if total_groups is not None and total_groups > 0:
        rank_pct = max(1, cur_rank) / float(total_groups)
        surge_delta = max(3, round(total_groups * 0.05))
        acc_delta = max(8, round(total_groups * 0.12))
        if rank_pct <= 0.20 and delta_1m >= surge_delta:
            return "🚀 Surging", "surging"
        elif 0.20 < rank_pct <= 0.60 and delta_1m >= acc_delta:
            return "🔄 Accumulating", "accumulating"
        elif rank_pct <= 0.25 and -surge_delta <= delta_1m <= surge_delta:
            return "⏸️ Consolidating", "consolidating"
        elif delta_1m <= -acc_delta:
            return "⚠️ Distributing", "distributing"
        elif rank_pct > 0.65:
            return "💤 Lagging", "lagging"
        else:
            return "⚪ Neutral", "neutral"

    if cur_rank <= 20 and delta_1m >= 5:
        return "🚀 Surging", "surging"
    elif 21 <= cur_rank <= 70 and delta_1m >= 15:
        return "🔄 Accumulating", "accumulating"
    elif cur_rank <= 25 and -5 <= delta_1m <= 5:
        return "⏸️ Consolidating", "consolidating"
    elif delta_1m <= -15:
        return "⚠️ Distributing", "distributing"
    elif cur_rank > 70:
        return "💤 Lagging", "lagging"
    else:
        return "⚪ Neutral", "neutral"


def classify_group_letter_grade(rank: int, total_groups: int) -> tuple[str, str, float, bool, bool]:
    """
    Classifies an industry group into standardized IBD Letter Grades (A+ to E) based on percentile rank:
    - A+: Top 10% (Percentile >= 90) -> Elite Institutional Leadership
    - A:  Next 10% (Percentile 80-89) -> Strong Leadership (Together with A+, forms Top 20%)
    - B:  Next 20% (Percentile 60-79) -> Favorable Accumulation
    - C:  Middle 20% (Percentile 40-59) -> Neutral / Market Perform
    - D:  Next 20% (Percentile 20-39) -> Lagging Distribution (O'Neil Rule: Avoid)
    - E:  Bottom 20% (Percentile < 20) -> Severe Laggards (O'Neil Rule: Never Buy)
    
    Returns:
        (grade_str, badge_style, percentile, is_top_20_pct, is_danger_group)
    """
    if total_groups <= 0:
        return "C", "neutral", 50.0, False, False
        
    rank_val = max(1, rank)
    rank_pct = (rank_val / total_groups) * 100.0
    percentile = round(((total_groups - rank_val + 1) / total_groups) * 100.0, 1)
    
    if rank_pct <= 10.0:
        return "A+", "grade_a_plus", percentile, True, False
    elif rank_pct <= 20.0:
        return "A", "grade_a", percentile, True, False
    elif rank_pct <= 40.0:
        return "B", "grade_b", percentile, False, False
    elif rank_pct <= 60.0:
        return "C", "grade_c", percentile, False, False
    elif rank_pct <= 80.0:
        return "D", "grade_d", percentile, False, True
    else:
        return "E", "grade_e", percentile, False, True

# 3-Horizon alignment: (label, bars) per horizon, and the excess-return band (in % points)
# inside which a horizon reads as "flat" (🟡) rather than out/under-performing.
HORIZON_WINDOWS = (("1M", 21), ("3M", 63), ("6M", 126))
HORIZON_FLAT_BAND = {"1M": 1.0, "3M": 2.0, "6M": 3.0}


@st.cache_data(ttl=3600, show_spinner=False)
def load_benchmark_close(universe_scope: str = "India (NSE/BSE)") -> tuple:
    """
    Returns (close_series, benchmark_name) for the broad-market benchmark.
    The local price cache holds constituents only, so the index is fetched separately.
    Returns (empty Series, name) if the download fails; callers fall back to the
    equal-weighted universe curve.
    """
    ticker, name = ("^CRSLDX", "NIFTY 500") if _scope_is_india(universe_scope) else ("^GSPC", "S&P 500")
    try:
        import yfinance as yf
        raw = yf.download(ticker, period="2y", progress=False, auto_adjust=True, threads=False)
        if raw is None or raw.empty:
            return pd.Series(dtype=float), name
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
        return close, name
    except Exception as e:
        print(f"[IndustryGroupEngine] Benchmark fetch failed for {ticker}: {e}")
        return pd.Series(dtype=float), name


def classify_horizon_alignment(excess: dict) -> tuple:
    """
    Classifies an industry's relative-strength agreement across 1M / 3M / 6M.
    excess: {'1M': x, '3M': y, '6M': z} = group return minus benchmark return (% points).
    Returns (badge, state) e.g. ("🟢🟡🔴", "🌱 Emerging Turn").

    - All three outperforming          -> 🟢 Confluence  (trend intact)
    - All three underperforming        -> 🔴 Chronic Laggard
    - Short-term up, long-term not up  -> 🌱 Emerging Turn (leadership turning up)
    - Short-term down, long-term up    -> ⚠️ Fading (leadership turning down)
    - Anything else                    -> 🔀 Mixed
    """
    marks = []
    for lbl, _ in HORIZON_WINDOWS:
        val = excess.get(lbl, np.nan)
        band = HORIZON_FLAT_BAND[lbl]
        if val is None or np.isnan(val):
            marks.append("⚪")
        elif val > band:
            marks.append("🟢")
        elif val < -band:
            marks.append("🔴")
        else:
            marks.append("🟡")
    badge = "".join(marks)
    short, _, long_ = marks

    if marks == ["🟢"] * 3:
        state = "🟢 Confluence"
    elif marks == ["🔴"] * 3:
        state = "🔴 Chronic Laggard"
    elif short == "🟢" and long_ in ("🔴", "🟡"):
        state = "🌱 Emerging Turn"
    elif short == "🔴" and long_ == "🟢":
        state = "⚠️ Fading"
    else:
        state = "🔀 Mixed"
    return badge, state


def compute_participation(df_matrix: pd.DataFrame) -> dict:
    """
    Participation ratio: share of ranked industries outperforming the benchmark per horizon.
    Headline uses 3M (medium term); 1M vs 3M shows whether breadth is broadening or narrowing.
    """
    out = {"total": 0, "benchmark": "Benchmark"}
    if df_matrix is None or df_matrix.empty or "Excess_3M" not in df_matrix.columns:
        return out
    out["total"] = len(df_matrix)
    out["benchmark"] = str(df_matrix["Benchmark_Name"].iloc[0]) if "Benchmark_Name" in df_matrix.columns else "Benchmark"
    for lbl, _ in HORIZON_WINDOWS:
        col = f"Excess_{lbl}"
        valid = df_matrix[col].dropna()
        above = int((valid > 0).sum())
        out[f"above_{lbl}"] = above
        out[f"pct_{lbl}"] = (above / len(valid) * 100.0) if len(valid) else 0.0

    pct = out["pct_3M"]
    if pct >= 55.0:
        out["reading"], out["tone"] = "Broad", "broad"
    elif pct >= 40.0:
        out["reading"], out["tone"] = "Selective", "selective"
    else:
        out["reading"], out["tone"] = "Narrow", "narrow"

    drift = out["pct_1M"] - out["pct_3M"]
    if drift >= 5.0:
        out["drift"] = "Broadening"
    elif drift <= -5.0:
        out["drift"] = "Narrowing"
    else:
        out["drift"] = "Stable"
    out["drift_pts"] = drift
    states = df_matrix["Horizon_State"] if "Horizon_State" in df_matrix.columns else pd.Series(dtype=str)
    out["confluence"] = int((states == "🟢 Confluence").sum())
    out["emerging"] = int((states == "🌱 Emerging Turn").sum())
    out["fading"] = int((states == "⚠️ Fading").sum())
    return out


def compute_dual_momentum_summary(df_matrix: pd.DataFrame) -> dict:
    """
    Computes Gary Antonacci Dual Momentum Regime metrics across the industry universe:
    Combines Relative Strength (Excess_3M vs Benchmark) with Absolute Momentum (Return_3M > 0).
    Separates True Alpha Compounders from Bear Market 'Relative Mirages'.
    """
    out = {
        "total": 0,
        "benchmark": "Benchmark",
        "bench_ret_3m": 0.0,
        "dual_alpha_count": 0,
        "dual_alpha_pct": 0.0,
        "mirage_count": 0,
        "mirage_pct": 0.0,
        "bleed_count": 0,
        "bleed_pct": 0.0,
        "beta_count": 0,
        "beta_pct": 0.0,
        "abs_1m_above": 0,
        "abs_1m_pct": 0.0,
        "abs_3m_above": 0,
        "abs_3m_pct": 0.0,
        "abs_6m_above": 0,
        "abs_6m_pct": 0.0,
        "mirage_spread": 0.0,
        "reading": "Selective Dual Momentum",
        "tone": "selective"
    }
    if df_matrix is None or df_matrix.empty or "Excess_3M" not in df_matrix.columns or "Return_3M" not in df_matrix.columns:
        return out
        
    n = len(df_matrix)
    out["total"] = n
    out["benchmark"] = str(df_matrix["Benchmark_Name"].iloc[0]) if "Benchmark_Name" in df_matrix.columns else "Benchmark"
    
    # Calculate implied benchmark 3M return
    diff = df_matrix["Return_3M"] - df_matrix["Excess_3M"]
    valid_diff = diff.dropna()
    if not valid_diff.empty:
        out["bench_ret_3m"] = float(valid_diff.iloc[0])
            
    # 4 Quadrants
    rel_pos = df_matrix["Excess_3M"] > 0
    abs_pos = df_matrix["Return_3M"] > 0
    
    alpha = int((rel_pos & abs_pos).sum())
    mirage = int((rel_pos & (~abs_pos)).sum())
    beta = int(((~rel_pos) & abs_pos).sum())
    bleed = int(((~rel_pos) & (~abs_pos)).sum())
    
    out["dual_alpha_count"] = alpha
    out["dual_alpha_pct"] = (alpha / n * 100.0) if n else 0.0
    out["mirage_count"] = mirage
    out["mirage_pct"] = (mirage / n * 100.0) if n else 0.0
    out["beta_count"] = beta
    out["beta_pct"] = (beta / n * 100.0) if n else 0.0
    out["bleed_count"] = bleed
    out["bleed_pct"] = (bleed / n * 100.0) if n else 0.0
    
    # Absolute returns across 1M, 3M, 6M
    if "Return_1M" in df_matrix.columns:
        a1 = int((df_matrix["Return_1M"] > 0).sum())
        out["abs_1m_above"] = a1
        out["abs_1m_pct"] = (a1 / n * 100.0) if n else 0.0
    if "Return_3M" in df_matrix.columns:
        a3 = int((df_matrix["Return_3M"] > 0).sum())
        out["abs_3m_above"] = a3
        out["abs_3m_pct"] = (a3 / n * 100.0) if n else 0.0
    if "Return_6M" in df_matrix.columns:
        a6 = int((df_matrix["Return_6M"] > 0).sum())
        out["abs_6m_above"] = a6
        out["abs_6m_pct"] = (a6 / n * 100.0) if n else 0.0
        
    # Mirage spread (Relative % - Dual %): how many groups are fake relative winners
    rel_above = int(rel_pos.sum())
    rel_pct = (rel_above / n * 100.0) if n else 0.0
    out["mirage_spread"] = rel_pct - out["dual_alpha_pct"]
    
    # Regime classification
    pct = out["dual_alpha_pct"]
    if pct >= 55.0 and out["mirage_count"] <= 10:
        out["reading"], out["tone"] = "Broad Dual Expansion", "broad"
    elif pct >= 40.0:
        out["reading"], out["tone"] = "Selective Dual Momentum", "selective"
    else:
        out["reading"], out["tone"] = "Defensive / Mirage Trap Risk", "narrow"
        
    return out


_KNOWN_DUAL_CLASS_ROOTS = {
    'GOOG': 'GOOGL', 'GOOGL': 'GOOGL',
    'FOX': 'FOXA', 'FOXA': 'FOXA',
    'NWS': 'NWSA', 'NWSA': 'NWSA',
    'WLY': 'WLYB', 'WLYB': 'WLYB',
    'LILA': 'LILAK', 'LILAK': 'LILAK',
    'CENT': 'CENTA', 'CENTA': 'CENTA',
    'Z': 'ZG', 'ZG': 'ZG',
    'UA': 'UAA', 'UAA': 'UAA',
    'PARA': 'PARAA', 'PARAA': 'PARAA',
    'DISCA': 'DISCK', 'DISCB': 'DISCK', 'DISCK': 'DISCK',
    'BRK-A': 'BRK-B', 'BRK-B': 'BRK-B', 'BRK.A': 'BRK-B', 'BRK.B': 'BRK-B',
    'BF-A': 'BF-B', 'BF-B': 'BF-B', 'BF.A': 'BF-B', 'BF.B': 'BF-B',
    'LEN': 'LEN', 'LEN-B': 'LEN', 'LEN.B': 'LEN',
    'HEI': 'HEI', 'HEI-A': 'HEI', 'HEI.A': 'HEI',
    'TAP': 'TAP', 'TAP-A': 'TAP',
    'BIO': 'BIO', 'BIO-B': 'BIO',
    'MKC': 'MKC', 'MKC-V': 'MKC',
    'STZ': 'STZ', 'STZ-B': 'STZ',
    'UHAL': 'UHAL', 'UHAL-B': 'UHAL',
    'MOG-A': 'MOG-A', 'MOG-B': 'MOG-A',
    'TATAMOTORS': 'TATAMOTORS', 'TATAMTRDVR': 'TATAMOTORS',
    'JAINIRR': 'JAINIRR', 'JISLDVREQS': 'JAINIRR',
    'GATECH': 'GATECH', 'GATECHDVR': 'GATECH',
}
_CLASS_SUFFIX_RE = re.compile(r"(?:[-.](?:A|B|C|V|DVR)|(?<=[A-Z]{3})DVR|DVREQS)$", re.IGNORECASE)
_CORP_NAME_STRIP_RE = re.compile(
    r"\b(?:class\s+[a-z0-9]+|series\s+[a-z0-9]+|dvr|inc\.?|incorporated|corp\.?|corporation|ltd\.?|limited|llc|plc|co\.?|company)\b",
    re.IGNORECASE,
)


def _share_class_root(ticker: str) -> str:
    base = str(ticker or "").strip().upper().replace(".NS", "").replace(".BO", "")
    base = _IN_SUFFIX_STRIP_RE.sub("", base)
    if base in _KNOWN_DUAL_CLASS_ROOTS:
        return _KNOWN_DUAL_CLASS_ROOTS[base]
    return _CLASS_SUFFIX_RE.sub("", base)


def _normalize_company_name_key(name: str) -> str:
    if not name:
        return ""
    cleaned = _CORP_NAME_STRIP_RE.sub(" ", str(name))
    cleaned = re.sub(r"[^A-Za-z0-9\s]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def _prepare_universe_price_panels(universe_scope: str = "India (NSE/BSE)") -> tuple:
    """
    Single shared price-hygiene & calendar-alignment pipeline for Matrix and Deep Dive:
      - Enforces strict India (.NS/.BO, NSE>BSE dedup) vs US scoping
      - Strips ETFs/indices and active_universe_set outsiders when populated
      - Trims NSE/NYSE exchange holiday rows consistently via min_sess threshold
    Returns (close_df, high_df, low_df, volume_df, core_benchmark_set).
    """
    res = load_price_history_matrix(include_volume=True, include_low=True)
    if len(res) == 4:
        close_df, high_df, low_df, volume_df = res
    elif len(res) == 3:
        close_df, high_df, volume_df = res
        low_df = None
    else:
        close_df, high_df = res
        low_df = None
        volume_df = pd.DataFrame(index=close_df.index, columns=close_df.columns).fillna(0.0)

    if close_df is None or close_df.empty:
        empty = pd.DataFrame()
        return empty, empty, None, empty, set()

    is_india = _scope_is_india(universe_scope)
    core_benchmark_set = set()
    active_universe_set = set()
    try:
        from universe_security_master import load_universe_tickers
        active_universe_set = set(load_universe_tickers("IN" if is_india else "US"))
    except Exception:
        active_universe_set = set()
        if not is_india:
            try:
                from us_universe_manager import get_broad_us_universe
                active_universe_set = set(get_broad_us_universe() or [])
            except Exception:
                pass
        if not active_universe_set:
            txt_file = "tickers.txt" if is_india else "tickers_us.txt"
            txt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), txt_file)
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, "r", encoding="utf-8") as _tf:
                        active_universe_set = {ln.strip() for ln in _tf if ln.strip() and not ln.strip().startswith("#")}
                except Exception:
                    pass

    from price_history_manager import is_excluded_etf_or_index
    min_valid_bars = min(10, max(2, len(close_df)))

    if is_india:
        try:
            from market_data import fetch_nifty_total_market_tickers
            core_benchmark_set = set(fetch_nifty_total_market_tickers(show_progress=False))
        except Exception:
            pass
        valid_cols = [
            c for c in close_df.columns
            if (c.endswith(".NS") or c.endswith(".BO"))
            and not is_excluded_etf_or_index(c)
            and close_df[c].dropna().shape[0] >= min_valid_bars
        ]
        if active_universe_set and len(set(valid_cols) & active_universe_set) >= min(50, max(1, len(active_universe_set) // 2)):
            valid_cols = [c for c in valid_cols if c in active_universe_set]

        ns_bases = {c.replace(".NS", "") for c in valid_cols if c.endswith(".NS")}
        valid_cols = [
            c for c in valid_cols
            if not (c.endswith(".BO") and c.replace(".BO", "") in ns_bases)
        ]
        min_sess = max(20, int(0.15 * len(valid_cols))) if len(valid_cols) >= 100 else 1
    else:
        try:
            from us_universe_manager import get_sp500_constituents, get_nasdaq100_constituents
            core_benchmark_set = set(get_sp500_constituents()).union(set(get_nasdaq100_constituents()))
        except Exception:
            pass
        valid_cols = [
            c for c in close_df.columns
            if not c.endswith((".NS", ".BO"))
            and not is_excluded_etf_or_index(c)
            and close_df[c].dropna().shape[0] >= min_valid_bars
        ]
        if active_universe_set and len(set(valid_cols) & active_universe_set) >= min(50, max(1, len(active_universe_set) // 2)):
            valid_cols = [c for c in valid_cols if c in active_universe_set]
        min_sess = max(50, int(0.15 * len(valid_cols))) if len(valid_cols) >= 150 else 1

    if not valid_cols:
        empty = pd.DataFrame()
        return empty, empty, None, empty, core_benchmark_set

    close_sub = close_df[valid_cols]
    close_sub = close_sub.loc[close_sub.notna().sum(axis=1) >= min_sess]
    high_cols = [c for c in valid_cols if c in high_df.columns]
    high_sub = high_df[high_cols].reindex(close_sub.index) if high_cols else close_sub
    low_sub = None
    if low_df is not None and not low_df.empty:
        low_cols = [c for c in valid_cols if c in low_df.columns]
        if low_cols:
            low_sub = low_df[low_cols].reindex(close_sub.index)
    vol_cols = [c for c in valid_cols if volume_df is not None and c in volume_df.columns]
    vol_sub = volume_df[vol_cols].reindex(close_sub.index) if vol_cols else pd.DataFrame(0.0, index=close_sub.index, columns=valid_cols)
    return close_sub, high_sub, low_sub, vol_sub, core_benchmark_set


def _load_db_enrichment_maps() -> tuple:
    """
    Batch-loads (mcap_map, funda_map, company_name_map) from Turso/SQLite via database.py.
    Keys are stored uppercase raw-ticker first to preserve India vs US isolation.
    """
    mcap_map = {}
    funda_map = {}
    company_name_map = {}
    try:
        from database import get_fundamentals_map_for_engine, read_sql_df

        raw_funda, _ = get_fundamentals_map_for_engine("ALL")
        for tk_u, info in (raw_funda or {}).items():
            if not isinstance(info, dict):
                continue
            funda_map[tk_u] = {
                "eps_growth": float(info.get("eps_growth", 0.0) or 0.0),
                "sales_growth": float(info.get("sales_growth", 0.0) or 0.0),
                "roe": float(info.get("roe", 0.0) or 0.0),
            }
            mc_val = info.get("market_cap")
            if mc_val is not None and float(mc_val or 0.0) > 0:
                mcap_map[tk_u] = float(mc_val)

        try:
            df_sm = read_sql_df(
                "SELECT yf_ticker, symbol, market, company_name, market_cap FROM security_master WHERE status = 'ACTIVE'"
            )
            if df_sm is not None and not df_sm.empty:
                for row_d in df_sm.to_dict(orient="records"):
                    key = str(row_d.get("yf_ticker") or row_d.get("symbol") or "").strip().upper()
                    if not key:
                        continue
                    cname = row_d.get("company_name")
                    if cname:
                        company_name_map[key] = str(cname).strip()
                    sm_cap = row_d.get("market_cap")
                    if sm_cap is not None and float(sm_cap or 0.0) > 0 and key not in mcap_map:
                        mcap_map[key] = float(sm_cap)
        except Exception:
            pass

        try:
            df_mc = read_sql_df("SELECT ticker, market_cap FROM market_cap_cache")
            if df_mc is not None and not df_mc.empty:
                for tk, mc in zip(df_mc["ticker"].tolist(), df_mc["market_cap"].tolist()):
                    if tk and mc is not None and float(mc or 0.0) > 0:
                        mcap_map.setdefault(str(tk).strip().upper(), float(mc))
        except Exception:
            pass
    except Exception:
        pass
    return mcap_map, funda_map, company_name_map


def _lookup_map_val(mapping: dict, ticker: str, default=None):
    tk_u = str(ticker or "").strip().upper()
    if tk_u in mapping:
        return mapping[tk_u]
    if tk_u.endswith((".NS", ".BO")):
        base = tk_u.rsplit(".", 1)[0]
        if base in mapping:
            return mapping[base]
    return default


def _dedupe_share_classes(
    tickers: list,
    stock_status_dict: dict,
    mcap_map: dict = None,
    company_name_map: dict = None,
) -> tuple:
    """
    Collapses NSE/BSE duplicates and dual-class shares of the same corporation inside an
    industry group (e.g. GOOG/GOOGL, FOX/FOXA, NWS/NWSA, LEN/LEN-B, HEI/HEI-A) so a single
    company is weighted and counted once.
    Retains the highest-ADTV share class as primary and returns (deduped_list, alias_to_primary).
    """
    if not tickers:
        return [], {}

    # Pass 1: Exchange-level (.NS > .BO) collapse
    seen_exch = {}
    for t in tickers:
        clean_t = str(t).replace(".NS", "").replace(".BO", "")
        if clean_t not in seen_exch or (t.endswith(".NS") and not seen_exch[clean_t].endswith(".NS")):
            seen_exch[clean_t] = t
    stage1 = list(seen_exch.values())
    if len(stage1) <= 1:
        return stage1, {}

    mcap_map = mcap_map or {}
    company_name_map = company_name_map or {}

    def _twin_bucket(tk: str) -> str:
        root_sym = _share_class_root(tk)
        cname = _normalize_company_name_key(_lookup_map_val(company_name_map, tk, ""))
        if cname and len(cname) >= 4:
            return f"name::{cname}"
        return f"sym::{root_sym}"

    # Group only share classes whose symbol roots match OR share identical normalized company name
    # + close market cap / symbol prefix (guards unrelated company-name noise).
    buckets = {}
    for tk in stage1:
        sym_key = _share_class_root(tk)
        cname = _normalize_company_name_key(_lookup_map_val(company_name_map, tk, ""))
        matched_key = None
        for bkey, members in buckets.items():
            rep = members[0]
            rep_sym = _share_class_root(rep)
            if sym_key == rep_sym:
                matched_key = bkey
                break
            rep_cname = _normalize_company_name_key(_lookup_map_val(company_name_map, rep, ""))
            if cname and rep_cname and cname == rep_cname:
                mc1 = float(_lookup_map_val(mcap_map, tk, 0.0) or 0.0)
                mc2 = float(_lookup_map_val(mcap_map, rep, 0.0) or 0.0)
                cap_close = (mc1 > 0 and mc2 > 0 and abs(mc1 - mc2) / max(mc1, mc2) <= 0.10)
                prefix_close = (len(sym_key) >= 2 and len(rep_sym) >= 2 and sym_key[:2] == rep_sym[:2])
                if cap_close or prefix_close:
                    matched_key = bkey
                    break
        if matched_key is None:
            matched_key = _twin_bucket(tk) + f"::{tk}"
            buckets[matched_key] = [tk]
        else:
            buckets[matched_key].append(tk)

    deduped = []
    alias_to_primary = {}
    for members in buckets.values():
        if len(members) == 1:
            deduped.append(members[0])
            continue
        best = max(
            members,
            key=lambda x: (
                bool(stock_status_dict.get(x, {}).get("is_tradable_leader", False)),
                float(stock_status_dict.get(x, {}).get("adtv_20", 0.0) or 0.0),
                int(stock_status_dict.get(x, {}).get("bars_count", 0) or 0),
                -len(str(x)),
            ),
        )
        deduped.append(best)
        for m in members:
            if m != best:
                alias_to_primary[m] = best

    return deduped, alias_to_primary


def _compute_smoothed_6m_rs_at_offset(
    close_clean: pd.DataFrame,
    tradable_cols_set: set,
    end_idx: int,
) -> tuple:
    """
    Vectorized MarketSurge 2.0 Smoothed 6-Month Constituent RS at bar `end_idx` (zero future leakage).
    Anchors use 3-bar averages at offsets 21/42/63/95/126 weighted 0.30/0.25/0.20/0.15/0.10.
    Cross-sectional 0-99 percentiles are computed strictly across active tradable stocks at `end_idx`,
    with non-tradable peers interpolated via np.searchsorted without shifting tradable ranks.
    Returns (rs_pct_series, raw_scores_series, active_tradable_set).
    """
    cols = close_clean.columns
    if close_clean.empty or end_idx < 1:
        zero_s = pd.Series(0, index=cols, dtype=int)
        nan_s = pd.Series(np.nan, index=cols, dtype=float)
        return zero_s, nan_s, set()

    cur_win = close_clean.iloc[max(0, end_idx - 2) : end_idx + 1]
    c_anchor = cur_win.mean(axis=0, skipna=True)
    # Require the stock to have valid closing bars within the trailing 5 sessions up to end_idx
    recent_notna = close_clean.iloc[max(0, end_idx - 4) : end_idx + 1].notna().any(axis=0)
    min_active_bars = min(21, max(5, end_idx))
    bars_up_to_end = close_clean.iloc[: end_idx + 1].notna().sum(axis=0)
    active_mask = recent_notna & (bars_up_to_end >= min_active_bars) & c_anchor.gt(0)

    anchors = ((21, 0.30), (42, 0.25), (63, 0.20), (95, 0.15), (126, 0.10))
    weighted_sum = pd.Series(0.0, index=cols, dtype=float)
    weight_acc = pd.Series(0.0, index=cols, dtype=float)

    for lookback, wt in anchors:
        t_a = end_idx - lookback
        if t_a < 0:
            continue
        win = close_clean.iloc[max(0, t_a - 1) : min(end_idx, t_a + 1) + 1]
        base = win.mean(axis=0, skipna=True)
        valid_h = active_mask & base.gt(0)
        ret = ((c_anchor / base.replace(0.0, np.nan)) - 1.0) * 100.0
        weighted_sum = weighted_sum + ret.where(valid_h, 0.0) * wt
        weight_acc = weight_acc + pd.Series(np.where(valid_h, wt, 0.0), index=cols)

    # Fallback for 5-20 bar synthetic mocks when end_idx < 21
    missing_h = active_mask & (weight_acc <= 0)
    if bool(missing_h.any()) and end_idx >= 5:
        t_a = max(0, end_idx - min(5, end_idx))
        base = close_clean.iloc[max(0, t_a - 1) : min(end_idx, t_a + 1) + 1].mean(axis=0, skipna=True)
        valid_fb = missing_h & base.gt(0)
        ret_fb = ((c_anchor / base.replace(0.0, np.nan)) - 1.0) * 100.0
        weighted_sum = weighted_sum.where(~valid_fb, ret_fb.where(valid_fb, 0.0))
        weight_acc = weight_acc.where(~valid_fb, 1.0)

    shrink_factor = (weight_acc / 0.65).clip(upper=1.0)
    raw_scores = ((weighted_sum / weight_acc.replace(0.0, np.nan)) * shrink_factor).where(active_mask & weight_acc.gt(0))

    trad_cols = [c for c in cols if c in tradable_cols_set]
    trad_scores = raw_scores.loc[trad_cols].dropna() if trad_cols else pd.Series(dtype=float)
    if trad_scores.empty:
        trad_scores = raw_scores.dropna()

    active_tradable_set = set(c for c in trad_cols if bool(active_mask.get(c, False)) and pd.notna(raw_scores.get(c)))
    rs_pct = pd.Series(0, index=cols, dtype=int)
    if trad_scores.empty:
        return rs_pct, raw_scores, active_tradable_set

    trad_ranks = (trad_scores.rank(pct=True) * 99.0).round(0).clip(0, 99).astype(int)
    rs_pct.loc[trad_ranks.index] = trad_ranks

    sorted_vals = np.sort(trad_scores.values.astype(float))
    non_trad_active = raw_scores.drop(labels=trad_scores.index, errors="ignore").dropna()
    if not non_trad_active.empty and len(sorted_vals) > 0:
        positions = np.searchsorted(sorted_vals, non_trad_active.values.astype(float), side="right")
        interp_pct = np.round((positions / float(len(sorted_vals))) * 99.0).clip(0, 99).astype(int)
        rs_pct.loc[non_trad_active.index] = interp_pct

    return rs_pct, raw_scores, active_tradable_set


def _build_group_curve(
    sub_close: pd.DataFrame,
    tradable_avail: list,
    mcap_map: dict,
    weight_mode: str = "capped_mcap",
    max_cap_w: float = 0.25,
) -> tuple:
    """
    Builds a chain-linked group price curve (base 100) from active non-bfill bars:
      - `capped_mcap`: reconstructs split-adjusted daily market cap cap_i(t) = (cap_today / close_today) * close_i(t)
        and iteratively redistributes excess above row_cap = max(max_cap_w, 1 / active_count) across uncapped members.
      - `equal`: equal-weighted active member returns.
    Uses lagged closing weights W.shift(1) for zero intra-bar lookahead.
    Returns (cum_curve_series, active_curve_bars).
    """
    sub = sub_close[tradable_avail]
    active_rows_mask = sub.notna().any(axis=1)
    active_curve_bars = int(active_rows_mask.sum())
    if active_curve_bars == 0:
        return pd.Series(dtype=float), 0

    first_valid_pos = int(np.argmax(active_rows_mask.values))
    sub = sub.iloc[first_valid_pos:]
    daily_rets = sub.pct_change(fill_method=None).clip(lower=-0.25, upper=0.25)

    if str(weight_mode).lower().strip() == "equal":
        mean_ret = daily_rets.mean(axis=1, skipna=True).fillna(0.0)
        cum_curve = (1.0 + mean_ret).cumprod() * 100.0
        return cum_curve, active_curve_bars

    # Dynamic split-adjusted market-cap weights
    known_caps = []
    for t in tradable_avail:
        mc = _lookup_map_val(mcap_map, t, 0.0)
        if mc is not None and float(mc) > 0:
            known_caps.append(float(mc))
    fallback_cap = float(np.median(known_caps)) if known_caps else 1.0

    c_last = sub.ffill().iloc[-1]
    scale_factors = {}
    for t in tradable_avail:
        mc = _lookup_map_val(mcap_map, t, 0.0)
        cap_today = float(mc) if (mc is not None and float(mc) > 0) else fallback_cap
        px_today = float(c_last.get(t, 0.0) or 0.0)
        scale_factors[t] = (cap_today / px_today) if px_today > 0 else 0.0

    cap_mat = sub.mul(pd.Series(scale_factors), axis=1)
    cap_mat = cap_mat.where(sub.notna() & (cap_mat > 0))
    row_totals = cap_mat.sum(axis=1, skipna=True).replace(0.0, np.nan)
    W = cap_mat.div(row_totals, axis=0).fillna(0.0)

    active_cnt = sub.notna().sum(axis=1).to_numpy()
    row_caps = np.maximum(float(max_cap_w), 1.0 / np.maximum(1, active_cnt))
    row_caps_s = pd.Series(row_caps, index=W.index)

    for _ in range(6):
        excess = W.sub(row_caps_s, axis=0).clip(lower=0.0)
        if float(excess.to_numpy(na_value=0.0).max()) <= 1e-6:
            break
        tot_excess = excess.sum(axis=1)
        capped_mask = W.ge(row_caps_s - 1e-9, axis=0)
        uncapped = W.where((~capped_mask) & sub.notna(), 0.0)
        uncapped_sum = uncapped.sum(axis=1)
        W = W.clip(upper=row_caps_s, axis=0)
        add = uncapped.div(uncapped_sum.replace(0.0, np.nan), axis=0).mul(tot_excess, axis=0).fillna(0.0)
        W = W + add

    W = W.div(W.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    W_prev = W.shift(1).where(sub.notna()).fillna(W)
    W_prev = W_prev.div(W_prev.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    grp_daily_ret = (W_prev * daily_rets.fillna(0.0)).sum(axis=1)
    cum_curve = (1.0 + grp_daily_ret).cumprod() * 100.0
    return cum_curve, active_curve_bars


def _get_smoothed_group_score_series(curves_df: pd.DataFrame, offset: int = 0) -> pd.Series:
    """
    5-Checkpoint MarketSurge 2.0 Smoothed 6-Month Group RS:
    3-day anchor returns at 21d/42d/63d/95d/126d weighted 0.30/0.25/0.20/0.15/0.10.
    """
    end_idx = len(curves_df) - 1 - offset
    if end_idx < 21:
        return pd.Series(index=curves_df.columns, dtype=float)

    c_now = curves_df.iloc[max(0, end_idx - 2) : end_idx + 1].mean(axis=0, skipna=True)
    anchors = ((21, 0.30), (42, 0.25), (63, 0.20), (95, 0.15), (126, 0.10))

    w_sum = pd.Series(0.0, index=curves_df.columns, dtype=float)
    w_tot = pd.Series(0.0, index=curves_df.columns, dtype=float)
    for lb, wt in anchors:
        t_a = end_idx - lb
        if t_a < 0:
            continue
        base = curves_df.iloc[max(0, t_a - 1) : min(end_idx, t_a + 1) + 1].mean(axis=0, skipna=True)
        valid = c_now.gt(0) & base.gt(0)
        ret = ((c_now / base.replace(0.0, np.nan)) - 1.0) * 100.0
        w_sum = w_sum + ret.where(valid, 0.0) * wt
        w_tot = w_tot + pd.Series(np.where(valid, wt, 0.0), index=curves_df.columns)

    return (w_sum / w_tot.replace(0.0, np.nan)).where(w_tot.gt(0))


def _compute_rsnh_panels(
    close_clean: pd.DataFrame,
    tradable_cols: list,
    bench_series: pd.Series = None,
) -> tuple:
    """
    Vectorized Ross Haber RS Line New High & O'Neil Pre-Breakout Blue Dot detector.
    Falls back cleanly to equal-weighted tradable universe benchmark if `bench_series` is missing.
    Guards recent IPOs with < min(126, max(40, int(win * 0.6))) bars while keeping `Tradable_Count` intact.
    Returns (is_rsnh_series, is_rs_blue_dot_series, is_near_52wh_series).
    """
    cols = close_clean.columns
    false_s = pd.Series(False, index=cols, dtype=bool)
    if close_clean.empty:
        return false_s, false_s, false_s

    win = min(252, len(close_clean))
    min_rsnh_bars = min(126, max(40, int(win * 0.6)))
    min_52wh_bars = min(63, max(20, int(win * 0.4)))

    if bench_series is not None and not bench_series.empty:
        bench_aligned = bench_series.reindex(close_clean.index, method="ffill").bfill()
    else:
        bench_aligned = pd.Series(dtype=float)

    if bench_aligned.dropna().empty or not bool((bench_aligned.dropna() > 0).all()):
        fallback_cols = [c for c in tradable_cols if c in close_clean.columns] or list(cols)
        fb_rets = (
            close_clean[fallback_cols]
            .pct_change(fill_method=None)
            .clip(lower=-0.25, upper=0.25)
            .mean(axis=1, skipna=True)
            .fillna(0.0)
        )
        bench_aligned = (1.0 + fb_rets).cumprod() * 100.0

    bench_aligned = bench_aligned.replace(0.0, np.nan)
    rs_line = close_clean.div(bench_aligned, axis=0)

    bars_s = close_clean.notna().sum(axis=0)
    c0 = close_clean.iloc[-1]
    stock_52w_high = close_clean.tail(win).max(axis=0, skipna=True)
    is_near_52wh = (bars_s >= min_52wh_bars) & c0.gt(0) & stock_52w_high.gt(0) & (c0 >= 0.85 * stock_52w_high)

    rs_win = rs_line.tail(win)
    rs_52w_high = rs_win.max(axis=0, skipna=True)
    rs_now = rs_line.iloc[-1]
    rs_recent_high = rs_line.tail(min(5, len(rs_line))).max(axis=0, skipna=True)
    rs_base_high = rs_line.tail(min(63, len(rs_line))).max(axis=0, skipna=True)

    is_rsnh = (bars_s >= min_rsnh_bars) & rs_now.gt(0) & rs_52w_high.gt(0) & (rs_recent_high >= 0.99 * rs_52w_high)
    in_constructive_base = c0.gt(0) & stock_52w_high.gt(0) & (c0 < 0.995 * stock_52w_high) & (c0 >= 0.75 * stock_52w_high)
    rs_leading_price = (
        rs_now.gt(0)
        & rs_base_high.gt(0)
        & (rs_now >= 0.99 * rs_base_high)
        & ((rs_now / rs_52w_high.replace(0.0, np.nan)) >= (c0 / stock_52w_high.replace(0.0, np.nan)) + 0.015)
    )
    is_blue_dot = (bars_s >= min_rsnh_bars) & in_constructive_base & (is_rsnh | rs_leading_price)
    return is_rsnh.fillna(False), is_blue_dot.fillna(False), is_near_52wh.fillna(False)


def _format_constit_leaders(
    grp: str,
    deduped_constits: list,
    stock_status_dict: dict,
    is_thematic: bool = False,
) -> tuple:
    """
    Sorts deduplicated constituents so clean non-short-history tradable leaders strictly precede
    recent IPOs and Tier-0 illiquids, then assigns deterministic Alpha/Sister/Pack/Sympathy roles.
    """
    def _leader_sort_key(t: str) -> tuple:
        # Priority 1: Tradable & Mature (not Tier-0 illiquid, not <63-bar IPO) AND RS >= 70 AND non-failed structure
        # Priority 2: Tradable & Mature AND RS >= 70 (even if failed structure, precedes RS < 70 laggards)
        # Priority 3: Tradable & Mature (all >=63-bar tradable stocks precede short-history IPOs)
        # Priority 4: Tradable (> Tier-0 floor precedes Tier-0 microcaps)
        # Priority 5: Effective Leadership Score (RS + bounded institutional liquidity & structural bonus)
        d = stock_status_dict.get(t, {})
        is_trad = bool(d.get("is_tradable_leader", False))
        is_sh = bool(d.get("is_short_history", False))
        inst_tier = int(d.get("inst_tier", 0))
        st = str(d.get("status", "⏳ BASE"))
        rs_v = int(d.get("rs", 0))
        trad_mature = is_trad and (not is_sh)
        rs70_gate = trad_mature and (rs_v >= 70)
        healthy_rs70 = rs70_gate and (st != "🔴 FAILED")
        tier_bonus = 3.0 if inst_tier >= 3 else (1.5 if inst_tier == 2 else -1.5)
        health_bonus = 2.0 if st in ("🟢 IN BUY ZONE", "🔵 RETEST", "⏳ COILING") else (-4.0 if st == "🔴 FAILED" else 0.0)
        eff_rs = float(rs_v) + tier_bonus + health_bonus
        return (
            healthy_rs70,
            rs70_gate,
            trad_mature,
            is_trad,
            eff_rs,
            rs_v,
            float(d.get("raw_comp", 0.0)),
            float(d.get("sortino_3m", 0.0)),
            float(d.get("adtv_20", 0.0)),
        )

    sorted_constits = sorted(deduped_constits, key=_leader_sort_key, reverse=True)

    top_leaders_full = []
    tradable_seen = 0
    for rank_idx, t in enumerate(sorted_constits):
        st_info = stock_status_dict.get(t, {})
        clean_t = re.sub(r"-(?:SM|ST|SZ|BE|BZ)$", "", t.replace(".NS", "").replace(".BO", ""))
        dist_p = st_info.get("dist_pivot", 0.0)
        sign = "+" if dist_p > 0 else ""
        dist_str = f"{sign}{dist_p:.1f}%"
        rs_val = int(st_info.get("rs", 0))
        is_trad = bool(st_info.get("is_tradable_leader", False))
        is_short = bool(st_info.get("is_short_history", False))

        if not is_trad:
            role_icon, role_code = "🚫", "TIER0_ILLIQUID"
            role_title = f"Tier-0 Illiquid (RS {rs_val})"
        elif len(sorted_constits) == 1:
            if rs_val >= 70:
                role_icon, role_code = "👑", "ALPHA"
                role_title = f"Solitary Leader (RS {rs_val})"
            else:
                role_icon, role_code = "—", "SOLITARY"
                role_title = f"Solitary Stock (RS {rs_val})"
        else:
            t_pos = tradable_seen
            tradable_seen += 1
            if t_pos == 0:
                if rs_val >= 80:
                    role_icon, role_code = "👑", "ALPHA"
                    role_title = f"Alpha Leader #1 (RS {rs_val})"
                elif rs_val >= 70:
                    role_icon, role_code = "👑", "ALPHA"
                    role_title = f"Leading Stock #1 (RS {rs_val})"
                else:
                    role_icon, role_code = "⚠️", "SYMPATHY"
                    role_title = f"Sympathy #1 (RS {rs_val})"
            elif t_pos == 1:
                if rs_val >= 75:
                    role_icon, role_code = "👯", "SISTER"
                    role_title = f"Sister Tandem #2 (RS {rs_val})"
                elif rs_val >= 70:
                    role_icon, role_code = "👯", "SISTER"
                    role_title = f"Tandem Contender #2 (RS {rs_val})"
                else:
                    role_icon, role_code = "⚠️", "SYMPATHY"
                    role_title = f"Sympathy #2 (RS {rs_val})"
            elif t_pos == 2:
                if rs_val >= 70:
                    role_icon, role_code = "🥉", "LEADER_3"
                    role_title = f"Leader #3 (RS {rs_val})" if rs_val >= 80 else f"Contender #3 (RS {rs_val})"
                else:
                    role_icon, role_code = "⚠️", "SYMPATHY"
                    role_title = f"Sympathy #3 (RS {rs_val})"
            elif rs_val >= 80:
                role_icon, role_code = "🐺", "PACK_LEADER"
                role_title = f"Pack Leader #{rank_idx + 1} (RS {rs_val})"
            else:
                role_icon, role_code = "⚠️", "SYMPATHY"
                role_title = f"Sympathy #{rank_idx + 1} (RS {rs_val})"

            if is_short and role_code != "TIER0_ILLIQUID":
                role_title = f"🆕 IPO · {role_title}"

        top_leaders_full.append({
            "rank": rank_idx + 1,
            "ticker": clean_t,
            "raw_ticker": t,
            "role_icon": role_icon,
            "role_code": role_code,
            "role_title": role_title,
            "rs": rs_val,
            "is_institutional": st_info.get("is_institutional", False),
            "is_tradable_leader": is_trad,
            "inst_tier": st_info.get("inst_tier", 0),
            "adtv_cr": st_info.get("adtv_cr", 0.0),
            "sortino_3m": st_info.get("sortino_3m", 0.0),
            "sortino_6m": st_info.get("sortino_6m", 0.0),
            "smooth_rs": st_info.get("smooth_rs", 0.0),
            "cmp": st_info.get("cmp", 0.0),
            "dist_pivot": dist_p,
            "dist_pivot_str": dist_str,
            "dist_ema": st_info.get("dist_ema", 0.0),
            "bars_count": st_info.get("bars_count", 63),
            "circuit_days_20d": st_info.get("circuit_days_20d", 0),
            "is_short_history": is_short,
            "is_rsnh": bool(st_info.get("is_rsnh", False)),
            "is_rs_blue_dot": bool(st_info.get("is_rs_blue_dot", False)),
            "status": st_info.get("status", "⏳ BASE"),
            "style": st_info.get("style", "base"),
        })

    top_3 = top_leaders_full[:3]
    leader_badges = []
    for ldr in top_3:
        st_lbl = ldr["status"]
        if "IN BUY ZONE" in st_lbl:
            badge_str = f"{ldr['ticker']} [🟢 BUY {ldr['dist_pivot_str']}]"
        elif "RETEST" in st_lbl:
            badge_str = f"{ldr['ticker']} [🔵 RETEST {ldr['dist_pivot_str']}]"
        elif "COILING" in st_lbl:
            badge_str = f"{ldr['ticker']} [⏳ COIL {ldr['dist_pivot_str']}]"
        elif "EXTENDED" in st_lbl:
            badge_str = f"{ldr['ticker']} [🟠 EXT {ldr['dist_pivot_str']}]"
        elif "FAILED" in st_lbl:
            badge_str = f"{ldr['ticker']} [🔴 FAIL]"
        else:
            badge_str = f"{ldr['ticker']} [⏳ BASE]"
        leader_badges.append(badge_str)

    return sorted_constits, top_leaders_full, top_3, "  •  ".join(leader_badges)


def load_cached_niche_industry_groups(universe_scope: str = "India (NSE/BSE)") -> pd.DataFrame:
    """Non-blocking mtime-keyed loader for persisted `.cache/industry_group_niche_*.json`."""
    _hydrate_igm_cache_from_turso_if_missing(universe_scope)
    is_in = _scope_is_india(universe_scope)
    niche_path = CACHE_NICHE_INDIA_PATH if is_in else CACHE_NICHE_US_PATH
    sig = _get_file_signature(niche_path)
    records = _load_json_payload_cached(niche_path, sig)
    if isinstance(records, list) and records:
        try:
            return pd.DataFrame(records)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


_MATRIX_RESULT_CACHE = {}


def _compute_capped_adtv_and_tradables(
    close_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df,
    volume_df: pd.DataFrame,
    core_benchmark_set: set,
    mcap_map: dict,
    is_india: bool,
) -> tuple:
    """Shared block-deal-capped ADTV, 63D hysteresis, circuit-lock & mcap tradability filter."""
    close_clean = close_df.ffill(limit=5)
    n_bars = len(close_clean)
    if n_bars == 0:
        return set(), {}, {
            "bars_count_s": pd.Series(dtype=int),
            "circuit_days_s": pd.Series(dtype=int),
            "min_price_floor": 10.0 if is_india else 2.00,
            "min_trad_bars": 5,
        }

    c0 = close_clean.iloc[-1]
    min_adtv_floor = 8_000_000.0 if is_india else 250_000.0
    min_price_floor = 10.0 if is_india else 2.00
    min_cap_floor = 500_000_000.0 if is_india else 50_000_000.0
    has_mcap_coverage = len(mcap_map or {}) >= 100

    has_volume_data = bool(volume_df is not None and not volume_df.empty and (volume_df > 0).any().any())
    if has_volume_data:
        vol_aligned = volume_df.reindex_like(close_df)
        turn_panel = close_df * vol_aligned
        turn_20 = turn_panel.tail(20)
        mean_20 = turn_20.mean(skipna=True).fillna(0.0)
        med_20 = turn_20.median(skipna=True).fillna(0.0)
        adtv_20_s = pd.Series(
            np.where(med_20 > 0, np.minimum(mean_20, med_20 * 2.0), mean_20),
            index=close_df.columns,
        ).fillna(0.0)
        med_63_s = turn_panel.tail(63).median(skipna=True).fillna(0.0)
    else:
        adtv_20_s = pd.Series(0.0, index=close_df.columns)
        med_63_s = pd.Series(0.0, index=close_df.columns)

    bars_count_s = close_df.notna().sum(axis=0)
    if low_df is not None and not low_df.empty and not high_df.empty:
        h_20 = high_df.tail(20)
        l_20 = low_df.tail(20)
        c_20_valid = close_df.tail(20).notna()
        is_circ = (((h_20 - l_20) / l_20.replace(0, np.nan)).abs() < 0.0005) & c_20_valid
        circuit_days_s = is_circ.sum(axis=0).astype(int)
    else:
        circuit_days_s = pd.Series(0, index=close_df.columns, dtype=int)

    c_20_tail = close_df.tail(20)
    c_20_mean = c_20_tail.mean(skipna=True).replace(0.0, np.nan)
    c_20_cov = (c_20_tail.std(skipna=True) / c_20_mean).fillna(1.0)
    c_20_diffs = (c_20_tail.diff().abs() > 1e-6).sum(axis=0)

    min_trad_bars = min(21, max(5, n_bars))
    tradable_universe_set = set()
    stock_adtv_map = {}
    core_set = core_benchmark_set or set()
    for t in close_df.columns:
        cur_cmp = float(c0.get(t, 0.0) or 0.0)
        cur_adtv = float(adtv_20_s.get(t, 0.0) or 0.0)
        med_63 = float(med_63_s.get(t, 0.0) or 0.0)
        stock_adtv_map[t] = cur_adtv
        n_b = int(bars_count_s.get(t, 0))
        circ_d = int(circuit_days_s.get(t, 0))

        price_ok = cur_cmp >= min_price_floor and np.isfinite(cur_cmp)
        bars_ok = n_b >= min_trad_bars
        circ_ok = circ_d <= 5
        clean_t = t.replace(".NS", "").replace(".BO", "")
        is_bench_t = (t in core_set or clean_t in core_set or f"{clean_t}.NS" in core_set)
        if has_volume_data:
            adtv_ok = is_bench_t or (cur_adtv >= min_adtv_floor) or (
                cur_adtv >= 0.75 * min_adtv_floor and med_63 >= 0.90 * min_adtv_floor
            )
            flat_ok = True if n_b < 21 else (float(c_20_cov.get(t, 1.0)) >= 0.002 and int(c_20_diffs.get(t, 3)) >= 3)
            mc = _lookup_map_val(mcap_map, t, 0.0)
            cap_ok = True
            if has_mcap_coverage and mc is not None and float(mc) > 0 and not is_bench_t:
                cap_ok = float(mc) >= min_cap_floor
            is_trad = price_ok and bars_ok and circ_ok and adtv_ok and flat_ok and cap_ok
        else:
            is_trad = price_ok and bars_ok and circ_ok

        if is_trad:
            tradable_universe_set.add(t)

    return tradable_universe_set, stock_adtv_map, {
        "bars_count_s": bars_count_s,
        "circuit_days_s": circuit_days_s,
        "min_price_floor": min_price_floor,
        "min_trad_bars": min_trad_bars,
    }


def _compute_matrix_core(
    close_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df,
    volume_df: pd.DataFrame,
    core_benchmark_set: set,
    group_map: dict,
    taxonomy: str,
    universe_scope: str,
    weight_mode: str,
    min_group_size: int,
    mcap_map: dict,
    funda_map: dict,
    company_name_map: dict,
) -> tuple:
    is_india = _scope_is_india(universe_scope)
    is_thematic = (str(taxonomy).lower().strip() == "thematic")
    close_clean = close_df.ffill(limit=5)
    high_clean = high_df.ffill(limit=5)

    n_bars = len(close_clean)
    if n_bars < 21 or not group_map:
        return pd.DataFrame(), pd.Series(), pd.DataFrame()

    c0 = close_clean.iloc[-1]
    tradable_universe_set, stock_adtv_map, _trad_meta = _compute_capped_adtv_and_tradables(
        close_df=close_df,
        high_df=high_df,
        low_df=low_df,
        volume_df=volume_df,
        core_benchmark_set=core_benchmark_set,
        mcap_map=mcap_map,
        is_india=is_india,
    )
    bars_count_s = _trad_meta["bars_count_s"]
    circuit_days_s = _trad_meta["circuit_days_s"]
    min_price_floor = _trad_meta["min_price_floor"]
    min_trad_bars = _trad_meta["min_trad_bars"]

    # Clean-universe Smoothed 6M Constituent RS at t = 0
    rs_percentile, composite_stock, _ = _compute_smoothed_6m_rs_at_offset(
        close_clean, tradable_universe_set, n_bars - 1
    )

    pivot_25 = high_clean.iloc[-26:-1].max() if len(high_clean) >= 27 else high_clean.max()
    ema_21 = close_clean.ewm(span=21, adjust=False).mean().iloc[-1]
    sma_50_s = close_clean.rolling(50, min_periods=20).mean().iloc[-1]
    sma_200_s = close_clean.rolling(200, min_periods=50).mean().iloc[-1]
    dist_pivot_s = ((c0 - pivot_25) / pivot_25.replace(0, np.nan)) * 100.0
    dist_ema_s = ((c0 - ema_21) / ema_21.replace(0, np.nan)) * 100.0

    # Benchmark series (loaded early so RSNH and Horizon Excess share one source)
    idx_close, idx_name = load_benchmark_close(universe_scope)
    bench_for_rsnh = pd.Series(dtype=float)
    if not idx_close.empty:
        aligned_bench = idx_close.reindex(close_clean.index, method="ffill").bfill()
        req_cov = min(127, len(aligned_bench))
        if len(aligned_bench) >= req_cov and aligned_bench.iloc[-req_cov:].notna().all():
            bench_for_rsnh = aligned_bench

    tradable_cols_list = [c for c in close_clean.columns if c in tradable_universe_set]
    is_rsnh_s, is_rs_bluedot_s, is_near_52wh_s = _compute_rsnh_panels(
        close_clean, tradable_cols_list, bench_series=bench_for_rsnh
    )

    # Mapped ticker risk metrics
    mar_rate = 0.065 if is_india else 0.045
    all_mapped_tickers = {t for tickers in group_map.values() for t in tickers if t in close_clean.columns}
    stock_sortino_map = {}
    for t in all_mapped_tickers:
        c_s = close_df[t].dropna()
        n_b = int(len(c_s))
        h_s = high_df[t].dropna() if t in high_df.columns else None
        l_s = low_df[t].dropna() if (low_df is not None and t in low_df.columns) else None
        if n_b >= 63:
            s3 = calculate_sortino_ratio(c_s, window=63, mar_annual=mar_rate)
        elif n_b >= 21:
            s3 = calculate_sortino_ratio(c_s, window=max(10, n_b - 1), mar_annual=mar_rate)
        else:
            s3 = 0.0
        s6 = calculate_sortino_ratio(c_s, window=126, mar_annual=mar_rate) if n_b >= 126 else 0.0
        sm_rs = calculate_smooth_rs_atr(c_s, h_s, l_s) if n_b >= 30 else 0.0
        stock_sortino_map[t] = (s3, s6, sm_rs, n_b)

    stock_status_dict = {}
    for t in close_clean.columns:
        p_dist = float(dist_pivot_s.get(t, np.nan))
        e_dist = float(dist_ema_s.get(t, np.nan))
        c_val = float(c0.get(t, np.nan))
        m50_v = float(sma_50_s.get(t, np.nan))
        m200_v = float(sma_200_s.get(t, np.nan))
        abv_50 = True if (np.isnan(c_val) or np.isnan(m50_v)) else bool(c_val >= m50_v)
        abv_200 = True if (np.isnan(c_val) or np.isnan(m200_v)) else bool(c_val >= m200_v)
        lbl, b_style, prio = classify_stock_action_state(p_dist, e_dist, above_50sma=abv_50, above_200sma=abv_200)
        s3, s6, sm_rs, n_b = stock_sortino_map.get(t, (0.0, 0.0, 0.0, int(bars_count_s.get(t, 0))))
        cur_cmp = float(c0.get(t, 0.0) or 0.0)
        cur_adtv = stock_adtv_map.get(t, 0.0)
        circ_d = int(circuit_days_s.get(t, 0))
        is_trad = t in tradable_universe_set
        clean_t = t.replace(".NS", "").replace(".BO", "")
        is_bench = (t in core_benchmark_set or clean_t in core_benchmark_set or f"{clean_t}.NS" in core_benchmark_set)

        if is_india:
            if cur_adtv >= 100_000_000.0 and cur_cmp >= min_price_floor and is_trad:
                inst_tier = 3
            elif cur_adtv >= 25_000_000.0 and cur_cmp >= min_price_floor and is_trad:
                inst_tier = 2
            elif is_trad:
                inst_tier = 1
            else:
                inst_tier = 0
        else:
            if cur_adtv >= 5_000_000.0 and cur_cmp >= min_price_floor and is_trad:
                inst_tier = 3
            elif cur_adtv >= 1_000_000.0 and cur_cmp >= min_price_floor and is_trad:
                inst_tier = 2
            elif is_trad:
                inst_tier = 1
            else:
                inst_tier = 0

        stock_status_dict[t] = {
            "status": lbl,
            "style": b_style,
            "priority": prio,
            "cmp": round(cur_cmp, 2),
            "adtv_20": round(cur_adtv, 1),
            "adtv_cr": round(cur_adtv / 1e7, 2) if is_india else round(cur_adtv / 1e6, 2),
            "is_institutional": bool(inst_tier >= 1),
            "is_tradable_leader": is_trad,
            "inst_tier": inst_tier,
            "is_benchmark_member": is_bench,
            "dist_pivot": round(p_dist, 1) if not np.isnan(p_dist) else 0.0,
            "dist_ema": round(e_dist, 1) if not np.isnan(e_dist) else 0.0,
            "rs": int(rs_percentile.get(t, 0)),
            "raw_comp": float(composite_stock.get(t, 0.0)) if pd.notna(composite_stock.get(t)) else 0.0,
            "sortino_3m": round(s3, 2),
            "sortino_6m": round(s6, 2),
            "smooth_rs": round(sm_rs, 2),
            "bars_count": int(n_b),
            "circuit_days_20d": circ_d,
            "is_short_history": bool(n_b < 126),
            "is_rsnh": bool(is_rsnh_s.get(t, False)),
            "is_rs_blue_dot": bool(is_rs_bluedot_s.get(t, False)),
            "is_near_52wh": bool(is_near_52wh_s.get(t, False)),
        }

    # Partition groups into Ranked Matrix (`>= min_group_size` tradables, excluded categories removed)
    # vs Niche (`1..min_group_size-1` or `Shell Companies` / `General / Unclassified`).
    excluded_groups = EXCLUDED_CANONICAL_GROUPS
    curves = {}
    ranked_meta = {}
    niche_rows = []
    min_curve_bars = min(130, max(45, n_bars))

    for grp, tickers in group_map.items():
        avail_raw = []
        for t in tickers:
            if t in close_clean.columns:
                avail_raw.append(t)
            elif f"{t}.NS" in close_clean.columns:
                avail_raw.append(f"{t}.NS")
            elif f"{t}.BO" in close_clean.columns:
                avail_raw.append(f"{t}.BO")
        avail_raw = [t for t in dict.fromkeys(avail_raw) if pd.notna(c0.get(t)) and float(c0.get(t, 0.0)) > 0]
        if not avail_raw:
            continue

        deduped_all, dual_alias_map = _dedupe_share_classes(
            avail_raw, stock_status_dict, mcap_map=mcap_map, company_name_map=company_name_map
        )
        tradable_avail = [t for t in deduped_all if stock_status_dict.get(t, {}).get("is_tradable_leader", False)]

        if grp in excluded_groups or len(tradable_avail) < min_group_size:
            sorted_c, top_full, top_3, ldr_disp = _format_constit_leaders(
                grp, deduped_all, stock_status_dict, is_thematic=is_thematic
            )
            t_cnt = len(tradable_avail)
            rsnh_cnt = sum(1 for t in tradable_avail if stock_status_dict.get(t, {}).get("is_rsnh", False))
            bd_cnt = sum(1 for t in tradable_avail if stock_status_dict.get(t, {}).get("is_rs_blue_dot", False))
            excl_reason = (
                "Excluded Category (Shell / Unclassified)"
                if grp in excluded_groups
                else f"Sub-Scale (<{min_group_size} Liquid Stocks: {t_cnt} Tradable)"
            )
            niche_cols = tradable_avail if tradable_avail else deduped_all
            niche_sub = close_clean[niche_cols] if niche_cols else pd.DataFrame()
            if not niche_sub.empty and len(niche_sub) > 21:
                n_c0 = niche_sub.iloc[-1]
                r_1w_n = round(float(((n_c0 / niche_sub.iloc[-6].replace(0, np.nan) - 1.0) * 100.0).mean(skipna=True) or 0.0), 2) if len(niche_sub) > 5 else 0.0
                r_1m_n = round(float(((n_c0 / niche_sub.iloc[-22].replace(0, np.nan) - 1.0) * 100.0).mean(skipna=True) or 0.0), 2)
                r_3m_n = round(float(((n_c0 / niche_sub.iloc[-64].replace(0, np.nan) - 1.0) * 100.0).mean(skipna=True) or 0.0), 2) if len(niche_sub) > 63 else 0.0
            else:
                r_1w_n, r_1m_n, r_3m_n = 0.0, 0.0, 0.0
            niche_rows.append({
                "Industry_Group": grp,
                "Exclusion_Reason": excl_reason,
                "Rank_Today": 999,
                "Group_Grade": "NR",
                "Group_Grade_Style": "neutral",
                "Group_Grade_Badge": "NR (Niche)" if grp not in excluded_groups else "NR (Excluded)",
                "Group_Percentile": 0.0,
                "Is_Top_20_Pct": False,
                "Is_Danger_Group": False,
                "Return_1W": float(r_1w_n if np.isfinite(r_1w_n) else 0.0),
                "Return_1M": float(r_1m_n if np.isfinite(r_1m_n) else 0.0),
                "Return_3M": float(r_3m_n if np.isfinite(r_3m_n) else 0.0),
                "Smoothed_6M_Score": 0.0,
                "Tradable_Count": t_cnt,
                "Unique_Company_Count": len(deduped_all),
                "Stock_Count": len(deduped_all),
                "RSNH_Count": rsnh_cnt,
                "RSNH_Pct": round((rsnh_cnt / t_cnt * 100.0), 1) if t_cnt > 0 else 0.0,
                "RS_BlueDot_Pct": round((bd_cnt / t_cnt * 100.0), 1) if t_cnt > 0 else 0.0,
                "Top_Leaders_Display": ldr_disp,
                "Top_3_Leaders": top_3,
                "Top_Leaders_Full": top_full,
                "Constituents": sorted_c,
                "Constituent_RS_Map": {t: stock_status_dict.get(t, {}).get("rs", 0) for t in sorted_c},
                "Dual_Class_Aliases": dual_alias_map,
                "Cache_Schema_Version": CACHE_SCHEMA_VERSION,
            })
            continue

        cap_w = 0.40 if len(tradable_avail) == 3 else (0.35 if len(tradable_avail) == 4 else 0.25)
        cum_curve, active_bars = _build_group_curve(
            close_df[tradable_avail], tradable_avail, mcap_map=mcap_map, weight_mode=weight_mode, max_cap_w=cap_w
        )
        if active_bars < min_curve_bars or cum_curve.empty:
            continue

        curves[grp] = cum_curve.reindex(close_clean.index)
        ranked_meta[grp] = {
            "deduped_all": deduped_all,
            "tradable_avail": tradable_avail,
            "dual_alias_map": dual_alias_map,
        }

    df_niche = pd.DataFrame(niche_rows)
    if not curves:
        return pd.DataFrame(), pd.Series(), df_niche

    curves_df = pd.DataFrame(curves, index=close_clean.index)
    # Forward-fill only internal gaps after each curve starts (zero pre-IPO bfill!)
    curves_df = curves_df.ffill()
    if curves_df.empty or len(curves_df) < min_curve_bars:
        return pd.DataFrame(), pd.Series(), df_niche

    benchmark_curve = curves_df.mean(axis=1, skipna=True).ffill().bfill()
    bench_name = "Equal-Wt Universe"
    bench_series = benchmark_curve
    if not bench_for_rsnh.empty:
        bench_series = bench_for_rsnh
        bench_name = idx_name

    bench_ret = {}
    for lbl, bars in HORIZON_WINDOWS:
        if len(bench_series) > bars and bench_series.iloc[-(bars + 1)] > 0:
            bench_ret[lbl] = float((bench_series.iloc[-1] / bench_series.iloc[-(bars + 1)] - 1) * 100.0)
        else:
            bench_ret[lbl] = np.nan

    # Precompute SMA-50 and Near-52WH panels once across entire universe for fast multi-offset breadth
    sma50_panel = close_clean.rolling(50, min_periods=20).mean()
    above_sma50_panel = close_clean > sma50_panel
    high252_panel = close_clean.rolling(252, min_periods=40).max()
    near_52wh_panel = (close_clean >= (0.85 * high252_panel)) & above_sma50_panel

    offset_defs = [("today", 0), ("1w", 5), ("3w", 15), ("1m", 21), ("6w", 30), ("3m", 63), ("6m", 126)]
    ranked_groups = list(curves_df.columns)
    M = len(ranked_groups)

    ranks_by_offset = {}
    blended_today = pd.Series(0.0, index=ranked_groups, dtype=float)
    grp_rs_pct_today = pd.Series(0.0, index=ranked_groups, dtype=float)
    breadth_pct_rank_today = pd.Series(0.0, index=ranked_groups, dtype=float)
    breadth_rs80_today = pd.Series(0.0, index=ranked_groups, dtype=float)
    breadth_sma50_today = pd.Series(0.0, index=ranked_groups, dtype=float)
    pure_price_rs = pd.Series(0, index=ranked_groups, dtype=int)
    pure_rs_rank = pd.Series(1, index=ranked_groups, dtype=int)

    for key, off in offset_defs:
        end_idx_t = n_bars - 1 - off
        if end_idx_t < 21:
            ranks_by_offset[key] = pd.Series(M, index=ranked_groups, dtype=int)
            continue

        grp_score_s = _get_smoothed_group_score_series(curves_df, offset=off).reindex(ranked_groups)
        valid_grp_scores = grp_score_s.dropna()
        if valid_grp_scores.empty:
            ranks_by_offset[key] = pd.Series(M, index=ranked_groups, dtype=int)
            continue

        if off == 0:
            rs_t, _, active_trad_t = rs_percentile, composite_stock, {
                t for t in tradable_universe_set if int(bars_count_s.get(t, 0)) >= min_trad_bars
            }
        else:
            rs_t, _, active_trad_t = _compute_smoothed_6m_rs_at_offset(
                close_clean, tradable_universe_set, end_idx_t
            )

        sma_row_t = above_sma50_panel.iloc[end_idx_t]
        near52_row_t = near_52wh_panel.iloc[end_idx_t]
        raw_b = {}
        b_rs80_map = {}
        b_sma_map = {}
        for g in ranked_groups:
            t_list = [t for t in ranked_meta[g]["tradable_avail"] if t in active_trad_t]
            denom = len(t_list)
            if denom == 0:
                raw_b[g] = 0.0
                b_rs80_map[g] = 0.0
                b_sma_map[g] = 0.0
            else:
                pack_t = sum(1 for t in t_list if int(rs_t.get(t, 0)) >= 80)
                sma_t = sum(1 for t in t_list if bool(sma_row_t.get(t, False)))
                n52_t = sum(1 for t in t_list if bool(near52_row_t.get(t, False)))
                b_80 = (pack_t / float(denom)) * 100.0
                b_50 = (sma_t / float(denom)) * 100.0
                b_near52 = (n52_t / float(denom)) * 100.0
                b_rs80_map[g] = b_80
                b_sma_map[g] = b_50
                raw_b[g] = 0.55 * b_80 + 0.25 * b_50 + 0.20 * b_near52

        grp_rs_pct = (grp_score_s.rank(pct=True) * 100.0).fillna(0.0)
        b_rank_pct = (pd.Series(raw_b).rank(pct=True) * 100.0).fillna(0.0)
        blended = (0.75 * grp_rs_pct) + (0.25 * b_rank_pct)

        if len(ranked_groups) >= 10 and end_idx_t >= 63:
            c_slice_t = curves_df[ranked_groups].iloc[:end_idx_t + 1]
            r1m_series = ((c_slice_t.iloc[-1] / c_slice_t.iloc[-22]) - 1.0) * 100.0 if len(c_slice_t) >= 22 else pd.Series(0.0, index=ranked_groups)
            r3m_series = ((c_slice_t.iloc[-1] / c_slice_t.iloc[-64]) - 1.0) * 100.0 if len(c_slice_t) >= 64 else pd.Series(0.0, index=ranked_groups)
            b3m_t = 0.0
            if bench_series is not None and len(bench_series) > end_idx_t and end_idx_t >= 63:
                b_now = float(bench_series.iloc[end_idx_t])
                b_prev = float(bench_series.iloc[end_idx_t - 63])
                if b_prev > 0:
                    b3m_t = ((b_now / b_prev) - 1.0) * 100.0
            damping = pd.Series(1.0, index=ranked_groups, dtype=float)
            for g_name in ranked_groups:
                r1_v = float(r1m_series.get(g_name, 0.0) or 0.0)
                r3_v = float(r3m_series.get(g_name, 0.0) or 0.0)
                b80_v = float(b_rs80_map.get(g_name, 0.0) or 0.0) if key == "today" else 10.0
                mult = 1.0
                if r3_v < 0.0 and r1_v <= 0.0:
                    # Negative over BOTH 1M and 3M: severe distribution penalty
                    mult *= 0.72
                elif r3_v < 0.0 and (r3_v - b3m_t) < 0.0:
                    # Dual Bleed (3M < 0% and 3M excess vs benchmark < 0%): moderate-strong damping
                    mult *= 0.84
                elif r3_v < 0.0:
                    # Relative Mirage (3M < 0%): mild damping
                    mult *= 0.92
                if key == "today" and b80_v <= 0.0:
                    # Zero RS>=80 stocks in group: cap ceiling
                    mult *= 0.88
                damping[g_name] = mult
            blended = blended * damping

        # Deterministic tie-free rank 1..M
        sort_order = sorted(
            ranked_groups,
            key=lambda g: (-float(blended.get(g, 0.0)), -float(grp_rs_pct.get(g, 0.0)), str(g)),
        )
        ranks_by_offset[key] = pd.Series({g: idx + 1 for idx, g in enumerate(sort_order)})

        if off == 0:
            blended_today = blended
            grp_rs_pct_today = grp_rs_pct
            breadth_pct_rank_today = b_rank_pct
            breadth_rs80_today = pd.Series(b_rs80_map)
            breadth_sma50_today = pd.Series(b_sma_map)
            pure_price_rs = (grp_score_s.rank(pct=True) * 99.0).fillna(0).round(0).clip(1, 99).astype(int)
            pure_order = sorted(
                ranked_groups,
                key=lambda g: (-float(grp_rs_pct.get(g, 0.0)), str(g)),
            )
            pure_rs_rank = pd.Series({g: idx + 1 for idx, g in enumerate(pure_order)})

    comp_rs_series = (blended_today.rank(pct=True) * 99.0).fillna(0).round(0).clip(1, 99).astype(int)
    cur_year = curves_df.index[-1].year
    ytd_bars = curves_df[curves_df.index.year == cur_year]

    def _funda_passes(t: str) -> bool:
        fn = _lookup_map_val(funda_map, t, None)
        if not isinstance(fn, dict):
            return False
        return float(fn.get("eps_growth", 0.0) or 0.0) >= 20.0 or float(fn.get("sales_growth", 0.0) or 0.0) >= 20.0

    rows = []
    for grp in ranked_groups:
        c_series = curves_df[grp].dropna()
        c_now = c_series.iloc[-1]
        r_1d = round(float((c_now / c_series.iloc[-2] - 1) * 100), 2) if len(c_series) > 1 and c_series.iloc[-2] > 0 else 0.0
        r_1w = round(float((c_now / c_series.iloc[-6] - 1) * 100), 2) if len(c_series) > 5 and c_series.iloc[-6] > 0 else 0.0
        r_1m = round(float((c_now / c_series.iloc[-22] - 1) * 100), 2) if len(c_series) > 21 and c_series.iloc[-22] > 0 else 0.0
        r_3m = round(float((c_now / c_series.iloc[-64] - 1) * 100), 2) if len(c_series) > 63 and c_series.iloc[-64] > 0 else 0.0
        r_6m = round(float((c_now / c_series.iloc[-127] - 1) * 100), 2) if len(c_series) > 126 and c_series.iloc[-127] > 0 else 0.0

        ytd_grp = ytd_bars[grp].dropna() if not ytd_bars.empty else pd.Series(dtype=float)
        if not ytd_grp.empty and ytd_grp.iloc[0] > 0:
            r_ytd = round(float((c_now / ytd_grp.iloc[0] - 1) * 100), 1)
        else:
            r_ytd = r_6m

        rank_td = int(ranks_by_offset["today"][grp])
        rank_w1 = int(ranks_by_offset["1w"][grp])
        rank_w3 = int(ranks_by_offset["3w"][grp])
        rank_m1 = int(ranks_by_offset["1m"][grp])
        rank_w6 = int(ranks_by_offset["6w"][grp])
        rank_m3 = int(ranks_by_offset["3m"][grp])
        rank_m6 = int(ranks_by_offset["6m"][grp])

        delta_1w = rank_w1 - rank_td
        delta_3w = rank_w3 - rank_td
        delta_1m = rank_m1 - rank_td
        delta_6w = rank_w6 - rank_td

        rot_status, rot_style = classify_rotation_status(rank_td, delta_1m, total_groups=M)
        grade_str, grade_style, grade_pct, is_top_20, is_danger = classify_group_letter_grade(rank_td, M)
        grade_badge = f"{grade_str} #{rank_td}"

        grp_ret = {
            "1M": r_1m if len(c_series) > 21 else np.nan,
            "3M": r_3m if len(c_series) > 63 else np.nan,
            "6M": r_6m if len(c_series) > 126 else np.nan,
        }
        excess = {lbl: grp_ret[lbl] - bench_ret[lbl] for lbl, _ in HORIZON_WINDOWS}
        horizon_badge, horizon_state = classify_horizon_alignment(excess)

        meta = ranked_meta[grp]
        tradable_avail = meta["tradable_avail"]
        deduped_all = meta["deduped_all"]
        dual_alias_map = meta["dual_alias_map"]

        t_cnt = len(tradable_avail)
        pack_count = sum(1 for t in tradable_avail if stock_status_dict.get(t, {}).get("rs", 0) >= 80)
        b_rs80_pct = round(float(breadth_rs80_today.get(grp, 0.0)), 1)
        b_sma50_pct = round(float(breadth_sma50_today.get(grp, 0.0)), 1)
        near_52wh_cnt = sum(1 for t in tradable_avail if stock_status_dict.get(t, {}).get("is_near_52wh", False))
        funda_cnt = sum(1 for t in tradable_avail if _funda_passes(t))
        rsnh_cnt = sum(1 for t in tradable_avail if stock_status_dict.get(t, {}).get("is_rsnh", False))
        bd_cnt = sum(1 for t in tradable_avail if stock_status_dict.get(t, {}).get("is_rs_blue_dot", False))

        near_52wh_pct = round((near_52wh_cnt / t_cnt) * 100.0, 1) if t_cnt > 0 else 0.0
        funda_pct = round((funda_cnt / t_cnt) * 100.0, 1) if t_cnt > 0 else 0.0
        rsnh_pct = round((rsnh_cnt / t_cnt) * 100.0, 1) if t_cnt > 0 else 0.0
        rs_bluedot_pct = round((bd_cnt / t_cnt) * 100.0, 1) if t_cnt > 0 else 0.0

        if b_rs80_pct >= 50.0 and pack_count >= 3:
            breadth_tier = "🔥 Broad Pack"
        elif pack_count >= 3 and b_rs80_pct >= 25.0:
            breadth_tier = "✅ Confirmed"
        else:
            breadth_tier = "⚠️ Narrow"

        grp_s3m = calculate_sortino_ratio(c_series, window=63, mar_annual=mar_rate)
        grp_s6m = calculate_sortino_ratio(c_series, window=126, mar_annual=mar_rate)
        grp_smooth_rs = calculate_smooth_rs_atr(c_series)
        apex_count = sum(
            1 for t in tradable_avail
            if stock_status_dict.get(t, {}).get("rs", 0) >= 80
            and stock_status_dict.get(t, {}).get("sortino_3m", 0.0) >= 3.0
        )

        sorted_constits, top_leaders_full, top_3, leader_display = _format_constit_leaders(
            grp, deduped_all, stock_status_dict, is_thematic=is_thematic
        )

        spark_vals = c_series.iloc[-21:]
        spark_norm = (spark_vals / spark_vals.iloc[0] * 100).round(1).tolist() if len(spark_vals) > 0 and spark_vals.iloc[0] > 0 else []

        is_rel_3m = bool(excess["3M"] > 0) if not np.isnan(excess["3M"]) else False
        is_abs_3m = bool(r_3m > 0) if not np.isnan(r_3m) else False
        if is_rel_3m and is_abs_3m:
            dual_state, dual_badge, dual_style = "🚀 True Dual Alpha", "🚀 Alpha", "alpha"
        elif is_rel_3m and not is_abs_3m:
            dual_state, dual_badge, dual_style = "🛡️ Relative Mirage", "🛡️ Mirage", "mirage"
        elif not is_rel_3m and is_abs_3m:
            dual_state, dual_badge, dual_style = "🔄 Beta Ride", "🔄 Beta", "beta"
        else:
            dual_state, dual_badge, dual_style = "📉 Dual Bleed", "📉 Bleed", "bleed"

        if is_thematic and t_cnt <= 3:
            tandem_count = sum(1 for t in tradable_avail if stock_status_dict.get(t, {}).get("rs", 0) >= 75)
            is_group_confirmed = bool(tandem_count >= 2)
            group_conf_badge = f"✅ Confirmed (Tandem {tandem_count}/{t_cnt})" if is_group_confirmed else f"⚪ Unconfirmed ({tandem_count}/2)"
        else:
            is_group_confirmed = bool(pack_count >= 3 and b_rs80_pct >= 25.0)
            group_conf_badge = f"✅ Confirmed ({pack_count} Leaders · {b_rs80_pct:.0f}%)" if is_group_confirmed else f"⚪ Unconfirmed ({pack_count} · {b_rs80_pct:.0f}%)"

        pack_sortinos = [
            stock_status_dict.get(t, {}).get("sortino_3m", 0.0)
            for t in tradable_avail
            if stock_status_dict.get(t, {}).get("rs", 0) >= 80
        ]
        if pack_sortinos:
            ldr_s3m = round(float(np.mean(sorted(pack_sortinos, reverse=True)[:10])), 2)
        else:
            top_c_s3 = [stock_status_dict.get(t, {}).get("sortino_3m", 0.0) for t in sorted_constits[:3]]
            ldr_s3m = round(float(np.mean(top_c_s3)), 2) if top_c_s3 else 0.0

        rows.append({
            "Industry_Group": grp,
            "Rank_Today": rank_td,
            "Group_Grade": grade_str,
            "Group_Grade_Style": grade_style,
            "Group_Grade_Badge": grade_badge,
            "Group_Percentile": grade_pct,
            "Is_Top_20_Pct": is_top_20,
            "Is_Danger_Group": is_danger,
            "Dual_State": dual_state,
            "Dual_Badge": dual_badge,
            "Dual_Style": dual_style,
            "Is_Dual_Alpha": bool(is_rel_3m and is_abs_3m),
            "Is_Relative_Mirage": bool(is_rel_3m and not is_abs_3m),
            "Is_Group_Confirmed": is_group_confirmed,
            "Group_Confirmation_Badge": group_conf_badge,
            "Rank_1W": rank_w1,
            "Rank_3W": rank_w3,
            "Rank_1M": rank_m1,
            "Rank_6W": rank_w6,
            "Rank_3M": rank_m3,
            "Rank_6M": rank_m6,
            "Delta_1W": delta_1w,
            "Delta_3W": delta_3w,
            "Delta_1M": delta_1m,
            "Delta_6W": delta_6w,
            "Rotation_Status": rot_status,
            "Rotation_Style": rot_style,
            "Horizon_Badge": horizon_badge,
            "Horizon_State": horizon_state,
            "Excess_1M": round(excess["1M"], 2) if not np.isnan(excess["1M"]) else np.nan,
            "Excess_3M": round(excess["3M"], 2) if not np.isnan(excess["3M"]) else np.nan,
            "Excess_6M": round(excess["6M"], 2) if not np.isnan(excess["6M"]) else np.nan,
            "Benchmark_Name": bench_name,
            "Comp_RS": int(comp_rs_series[grp]),
            "Pure_RS_Rank": int(pure_rs_rank[grp]),
            "Pure_Price_RS": int(pure_price_rs[grp]),
            "Blended_Score": round(float(blended_today[grp]), 2),
            "Breadth_Pct_Rank": round(float(breadth_pct_rank_today[grp]), 2),
            "Breadth_RS80_Pct": b_rs80_pct,
            "Breadth_50SMA_Pct": b_sma50_pct,
            "Breadth_Tier": breadth_tier,
            "Near_52WH_Pct": near_52wh_pct,
            "CANSLIM_Funda_Pct": funda_pct,
            "RSNH_Count": rsnh_cnt,
            "RSNH_Pct": rsnh_pct,
            "RS_BlueDot_Pct": rs_bluedot_pct,
            "Weight_Mode": str(weight_mode),
            "Pack_Hunting_Count": pack_count,
            "Tradable_Count": t_cnt,
            "Unique_Company_Count": len(deduped_all),
            "Stock_Count": len(deduped_all),
            "Sortino_3M": round(grp_s3m, 2),
            "Sortino_6M": round(grp_s6m, 2),
            "Leader_Sortino_3M": ldr_s3m,
            "Smooth_RS": round(grp_smooth_rs, 2),
            "Apex_Leaders": apex_count,
            "Return_1D": r_1d,
            "Return_1W": r_1w,
            "Return_1M": r_1m,
            "Return_3M": r_3m,
            "Return_6M": r_6m,
            "Return_YTD": r_ytd,
            "Top_Leaders_Display": leader_display,
            "Top_3_Leaders": top_3,
            "Top_Leaders_Full": top_leaders_full,
            "Sparkline_1M": spark_norm,
            "Constituents": sorted_constits,
            "Constituent_RS_Map": {t: stock_status_dict.get(t, {}).get("rs", 0) for t in sorted_constits},
            "Dual_Class_Aliases": dual_alias_map,
            "Cache_Schema_Version": CACHE_SCHEMA_VERSION,
        })

    df_result = pd.DataFrame(rows).sort_values("Rank_Today", ascending=True).reset_index(drop=True)
    return df_result, benchmark_curve, df_niche


def _hydrate_igm_cache_from_turso_if_missing(universe_scope: str = "India (NSE/BSE)") -> dict | None:
    """Hydrate `.cache/industry_group_{matrix,niche,intelligence}_*.json` once from Turso on empty Cloud disk."""
    is_in = _scope_is_india(universe_scope)
    intel_path = CACHE_INTEL_INDIA_PATH if is_in else CACHE_INTEL_US_PATH
    matrix_path = CACHE_MATRIX_INDIA_PATH if is_in else CACHE_MATRIX_US_PATH
    niche_path = CACHE_NICHE_INDIA_PATH if is_in else CACHE_NICHE_US_PATH

    if os.path.exists(matrix_path) and os.path.exists(niche_path) and os.path.exists(intel_path):
        return None

    cloud_key = "igm_bundle_in" if is_in else "igm_bundle_us"
    try:
        from database import load_cloud_json_cache_cached
        bundle, _ = load_cloud_json_cache_cached(cloud_key)
    except Exception:
        bundle = None

    if not isinstance(bundle, dict):
        return None

    df_matrix = bundle.get("matrix_df")
    if not isinstance(df_matrix, pd.DataFrame) or df_matrix.empty:
        return None
    df_niche = bundle.get("niche_df")
    if not isinstance(df_niche, pd.DataFrame):
        df_niche = pd.DataFrame()

    stock_intel = _build_stock_intelligence_dict(df_matrix, df_niche=df_niche)
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        _write_atomic_json(intel_path, stock_intel)
        _write_atomic_json(matrix_path, df_matrix.to_dict(orient="records"))
        niche_records = df_niche.to_dict(orient="records") if not df_niche.empty else []
        _write_atomic_json(niche_path, niche_records)
        try:
            _load_json_payload_cached.clear()
        except Exception:
            pass
    except Exception as e:
        print(f"[IndustryGroupEngine] Turso hydration disk write notice: {e}")
    return bundle


def _load_turso_igm_bundle(taxonomy: str, universe_scope: str) -> tuple:
    """Return (df_matrix, benchmark_curve, df_niche) from Turso cloud_json_cache when local .pkl is absent."""
    is_in = _scope_is_india(universe_scope)
    is_thematic = str(taxonomy).lower().strip() == "thematic"
    if is_thematic:
        cloud_key = "igm_thematic_in"
    else:
        cloud_key = "igm_bundle_in" if is_in else "igm_bundle_us"
    try:
        from database import load_cloud_json_cache_cached
        bundle, _ = load_cloud_json_cache_cached(cloud_key)
    except Exception:
        bundle = None
    if not isinstance(bundle, dict):
        return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame()
    df_m = bundle.get("matrix_df")
    b_c = bundle.get("benchmark_curve")
    df_n = bundle.get("niche_df")
    return (
        df_m.copy() if isinstance(df_m, pd.DataFrame) else pd.DataFrame(),
        b_c.copy() if isinstance(b_c, pd.Series) else pd.Series(dtype=float),
        df_n.copy() if isinstance(df_n, pd.DataFrame) else pd.DataFrame(),
    )


def compute_industry_group_matrix_with_niche(
    taxonomy: str = "canonical",
    universe_scope: str = "India (NSE/BSE)",
    weight_mode: str = "capped_mcap",
    min_group_size: int = None,
) -> tuple:
    """
    MarketSurge 2.0 O'Neil-Grade Matrix Engine.
    Returns (df_matrix, benchmark_curve, df_niche), separating `< min_group_size` groups
    so ranked denominators stay 100% pure.
    """
    _hydrate_igm_cache_from_turso_if_missing(universe_scope)
    close_df, high_df, low_df, volume_df, core_benchmark_set = _prepare_universe_price_panels(universe_scope)
    if close_df.empty:
        return _load_turso_igm_bundle(taxonomy, universe_scope)

    group_map = _resolve_group_map(taxonomy=taxonomy, universe_scope=universe_scope)
    if not group_map:
        return pd.DataFrame(), pd.Series(), pd.DataFrame()

    is_thematic = (str(taxonomy).lower().strip() == "thematic")
    eff_min_size = int(min_group_size) if min_group_size is not None else (
        MIN_RANKED_GROUP_SIZE_THEMATIC if is_thematic else MIN_RANKED_GROUP_SIZE_CANONICAL
    )
    eff_weight_mode = str(weight_mode or "capped_mcap").lower().strip()

    db_sig = _get_db_cache_sig()
    last_row_sum = round(float(close_df.iloc[-1].sum(skipna=True)), 4) if not close_df.empty else 0.0
    grp_sig = (len(group_map), tuple(sorted(group_map.keys())[:5]), sum(len(v) for v in group_map.values()))
    cache_key = (
        str(taxonomy),
        _scope_is_india(universe_scope),
        eff_weight_mode,
        eff_min_size,
        close_df.shape,
        str(close_df.index[-1]),
        tuple(close_df.columns[:5]),
        last_row_sum,
        grp_sig,
        db_sig,
    )
    cached = _MATRIX_RESULT_CACHE.get(cache_key)
    if cached is not None:
        df_m, b_c, df_n = cached
        return df_m.copy(), b_c.copy(), df_n.copy()

    mcap_map, funda_map, company_name_map = _load_db_enrichment_maps()
    df_matrix, bench_curve, df_niche = _compute_matrix_core(
        close_df=close_df,
        high_df=high_df,
        low_df=low_df,
        volume_df=volume_df,
        core_benchmark_set=core_benchmark_set,
        group_map=group_map,
        taxonomy=taxonomy,
        universe_scope=universe_scope,
        weight_mode=eff_weight_mode,
        min_group_size=eff_min_size,
        mcap_map=mcap_map,
        funda_map=funda_map,
        company_name_map=company_name_map,
    )
    if len(_MATRIX_RESULT_CACHE) >= 6:
        _MATRIX_RESULT_CACHE.pop(next(iter(_MATRIX_RESULT_CACHE)))
    _MATRIX_RESULT_CACHE[cache_key] = (df_matrix.copy(), bench_curve.copy(), df_niche.copy())
    return df_matrix, bench_curve, df_niche


def compute_industry_group_matrix(
    taxonomy: str = "canonical",
    universe_scope: str = "India (NSE/BSE)",
    weight_mode: str = "capped_mcap",
    min_group_size: int = None,
):
    """
    Computes MarketSurge 2.0 industry group relative strength matrix across Today..6M.
    Returns (df_matrix, benchmark_curve) preserving exact 2-tuple compatibility.
    """
    df_matrix, benchmark_curve, _ = compute_industry_group_matrix_with_niche(
        taxonomy=taxonomy,
        universe_scope=universe_scope,
        weight_mode=weight_mode,
        min_group_size=min_group_size,
    )
    return df_matrix, benchmark_curve


compute_industry_group_matrix_with_niche.clear = _MATRIX_RESULT_CACHE.clear
compute_industry_group_matrix.clear = _MATRIX_RESULT_CACHE.clear

def get_unranked_stock_group_payload(ticker: str, total_groups: int = 111) -> dict:
    """Standardized fallback for Day-0 IPOs or unclassified assets."""
    clean_t = ticker.replace('.NS', '').replace('.BO', '')
    return {
        'ticker': clean_t,
        'raw_ticker': ticker,
        'group_name': 'General / Unclassified',
        'group_rank': 999,
        'total_groups': total_groups,
        'group_grade': 'NR',
        'group_grade_style': 'neutral',
        'group_grade_badge': 'NR (Unclassified)',
        'is_top_20_pct': False,
        'is_danger_group': False,
        'stock_role': '— Unranked Constituent',
        'role_code': 'UNRANKED',
        'group_rotation_status': '⚪ Neutral',
        'group_delta_1m': 0,
        'alpha_leader': 'None',
        'sister_stock': 'None',
        'is_group_confirmed': False,
        'group_confirmation_badge': '⚪ Unranked'
    }

def _register_stock_intel_aliases(stock_intel: dict, raw_ticker: str, clean_t: str, item: dict) -> None:
    """Register clean, exchange-suffixed, and SME ticker aliases for fast O(1) lookup."""
    stock_intel[clean_t] = item
    stock_intel[raw_ticker] = item

    if raw_ticker.endswith('.NS') or raw_ticker.endswith('.BO'):
        stock_intel[f"{clean_t}.NS"] = item
        stock_intel[f"{clean_t}.BO"] = item

    if any(sfx in clean_t for sfx in ('-SM', '-ST', '-BE', '-BZ')):
        for sfx in ('-SM', '-ST', '-BE', '-BZ'):
            if sfx in clean_t:
                base_name = clean_t.replace(sfx, '')
                stock_intel[base_name] = item
                stock_intel[f"{base_name}.NS"] = item
                stock_intel[f"{base_name}.BO"] = item
    elif clean_t in {'ANONDITA', 'BRANDMAN', 'CLASSICEIL', 'MONOLITH', 'OBSCP', 'PARTH', 'PRIZOR', 'SHETHJI', 'VMARCIND'}:
        for sfx in ('-SM.NS', '-ST.NS', '-SM', '-ST'):
            stock_intel[f"{clean_t}{sfx}"] = item


def _build_intel_from_row(stock_intel: dict, row, total_groups: int, is_niche: bool = False) -> None:
    """Populate stock intelligence dictionary entries for one ranked or niche group row."""
    grp = row['Industry_Group']
    rank = 999 if is_niche else int(row['Rank_Today'])
    grade = 'NR' if is_niche else row.get('Group_Grade', 'C')
    grade_style = 'neutral' if is_niche else row.get('Group_Grade_Style', 'neutral')
    grade_badge = f"NR (Niche <{MIN_RANKED_GROUP_SIZE_CANONICAL})" if is_niche else row.get('Group_Grade_Badge', f"{grade} #{rank}")
    is_top_20 = False if is_niche else bool(row.get('Is_Top_20_Pct', False))
    is_danger = False if is_niche else bool(row.get('Is_Danger_Group', False))
    rot_status = row.get('Rotation_Status', '⚪ Niche / Micro')
    delta_1m = int(row.get('Delta_1M', 0))
    is_grp_conf = False if is_niche else bool(row.get('Is_Group_Confirmed', False))
    grp_conf_badge = '⚪ Niche / Unranked' if is_niche else row.get('Group_Confirmation_Badge', '⚪ Unconfirmed')

    constits = row.get('Constituents', [])
    rs_map = row.get('Constituent_RS_Map', {})
    top_3 = row.get('Top_3_Leaders', [])
    top_full = row.get('Top_Leaders_Full', []) or top_3

    ldr_meta_by_raw = {str(l.get('raw_ticker', '')): l for l in top_full if isinstance(l, dict)}
    tradable_list = [l for l in top_full if isinstance(l, dict) and l.get('is_tradable_leader', l.get('is_tradable', True))]
    tradable_pos = {str(l.get('raw_ticker', '')): idx for idx, l in enumerate(tradable_list)}
    alpha_ticker = (
        tradable_list[0]['ticker']
        if (not is_danger and len(tradable_list) > 0 and tradable_list[0].get('role_code') == 'ALPHA')
        else 'None'
    )
    sister_ticker = (
        tradable_list[1]['ticker']
        if (not is_danger and len(tradable_list) > 1 and tradable_list[1].get('role_code') == 'SISTER')
        else 'None'
    )

    for t in constits:
        clean_t = t.replace('.NS', '').replace('.BO', '')
        ldr_meta = ldr_meta_by_raw.get(t, {})
        rs_val = int(ldr_meta.get('rs', rs_map.get(t, 0)))
        is_tradable = bool(ldr_meta.get('is_tradable_leader', ldr_meta.get('is_tradable', True))) and (
            ldr_meta.get('role_code') != 'TIER0_ILLIQUID'
        )

        if is_niche:
            role = "— Niche Group Member"
            role_code = "NICHE"
        elif not is_tradable:
            role = "🚫 Illiquid / Low-Vol"
            role_code = "ILLIQUID"
        elif is_danger:
            t_idx = tradable_pos.get(t, 999)
            if rs_val >= 75:
                role = f"🐺 Pack Leader (Lagging Grp · RS {rs_val})"
                role_code = "PACK_LEADER"
            else:
                role = f"⚠️ Lagging Group #{t_idx + 1} (Avoid)"
                role_code = "LAGGARD_AVOID"
        else:
            rc = str(ldr_meta.get('role_code') or 'SYMPATHY')
            rt = str(ldr_meta.get('role_title') or '')
            ri = str(ldr_meta.get('role_icon') or '')
            if rc in ('ALPHA', 'SISTER', 'LEADER_3', 'PACK_LEADER', 'SOLITARY'):
                role_code = rc
                role = f"{ri} {rt}".strip()
            else:
                t_idx = tradable_pos.get(t, 999)
                role_code = "SYMPATHY"
                role = f"⚠️ Sympathy #{t_idx + 1} (Avoid)"

        item = {
            'ticker': clean_t,
            'raw_ticker': t,
            'group_name': grp,
            'group_rank': rank,
            'total_groups': total_groups,
            'group_grade': grade,
            'group_grade_style': grade_style,
            'group_grade_badge': grade_badge,
            'is_top_20_pct': is_top_20,
            'is_danger_group': is_danger,
            'stock_role': role,
            'role_code': role_code,
            'group_rotation_status': rot_status,
            'group_delta_1m': delta_1m,
            'alpha_leader': alpha_ticker,
            'sister_stock': sister_ticker,
            'is_group_confirmed': is_grp_conf,
            'group_confirmation_badge': grp_conf_badge,
        }
        _register_stock_intel_aliases(stock_intel, t, clean_t, item)

    # Also point excluded dual-class share twins to their canonical twin's intel payload
    for twin_raw, canon_raw in (row.get('Dual_Class_Aliases', {}) or {}).items():
        canon_clean = str(canon_raw).replace('.NS', '').replace('.BO', '')
        canon_item = stock_intel.get(canon_raw) or stock_intel.get(canon_clean)
        if canon_item:
            twin_clean = str(twin_raw).replace('.NS', '').replace('.BO', '')
            _register_stock_intel_aliases(stock_intel, twin_raw, twin_clean, canon_item)


def _build_stock_intelligence_dict(df_matrix: pd.DataFrame, df_niche: pd.DataFrame = None) -> dict:
    stock_intel = {}
    total_groups = len(df_matrix) if df_matrix is not None else 0

    if df_matrix is not None and not df_matrix.empty:
        for _, row in df_matrix.iterrows():
            _build_intel_from_row(stock_intel, row, total_groups, is_niche=False)

    if df_niche is not None and not df_niche.empty:
        for _, row in df_niche.iterrows():
            _build_intel_from_row(stock_intel, row, total_groups, is_niche=True)

    return stock_intel

def _scope_is_india(universe_scope: str = "India (NSE/BSE)") -> bool:
    """Return True for Indian market scopes and False for US scopes."""
    sc = str(universe_scope or "India").strip().upper()
    return not (sc.startswith("US") or sc == "US_MARKET")


@st.cache_data(ttl=3600, show_spinner=False)
def _load_json_payload_cached(cache_path: str, cache_sig: tuple):
    """Mtime-keyed pure local disk JSON reader (<10ms)."""
    if not cache_sig or cache_sig == (0, 0) or not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def extract_tradable_top_leaders(
    group_row_or_list,
    max_n: int = 3,
    universe_tickers_set: set = None,
    min_adtv: float = None,
    min_bars: int = 60,
    fallback_to_exchange: bool = True,
    min_rs: float = None,
) -> dict:
    """
    Derive clean, liquid institutional Top-N leaders from cached `Top_Leaders_Full`
    while separating raw short-history/illiquid Exchange #1 outliers (e.g. <60 bars or
    microcap ADTV traps).

    When `universe_tickers_set` (e.g. NIFTY 750) yields 0 matches for a niche group and
    `fallback_to_exchange=True`, falls back cleanly to Exchange Tradable Leaders and sets
    `is_universe_fallback=True` so UI cards never render empty `-` cells.

    Returns:
        {
            'tradable_leaders': list[dict],
            'alpha': dict | None,
            'sister': dict | None,
            'leader_3': dict | None,
            'raw_alpha': dict | None,
            'raw_alpha_anomaly': bool,
            'is_universe_fallback': bool,
        }
    """
    if isinstance(group_row_or_list, dict):
        leaders_raw = (
            group_row_or_list.get("Top_Leaders_Full")
            or group_row_or_list.get("Top_3_Leaders")
            or []
        )
    elif hasattr(group_row_or_list, "get"):
        leaders_raw = (
            group_row_or_list.get("Top_Leaders_Full")
            or group_row_or_list.get("Top_3_Leaders")
            or []
        )
    elif isinstance(group_row_or_list, list):
        leaders_raw = group_row_or_list
    else:
        leaders_raw = []

    if not isinstance(leaders_raw, list) or not leaders_raw:
        return {
            "tradable_leaders": [],
            "alpha": None,
            "sister": None,
            "leader_3": None,
            "raw_alpha": None,
            "raw_alpha_anomaly": False,
            "is_universe_fallback": False,
        }

    norm_uni = None
    if universe_tickers_set:
        norm_uni = set()
        for ut in universe_tickers_set:
            u_up = str(ut or "").strip().upper()
            if not u_up:
                continue
            norm_uni.add(u_up)
            u_clean = u_up.replace(".NS", "").replace(".BO", "")
            u_clean = _IN_SUFFIX_STRIP_RE.sub("", u_clean)
            norm_uni.add(u_clean)

    def _in_universe(ldr: dict) -> bool:
        if norm_uni is None:
            return True
        raw_t = str(ldr.get("raw_ticker") or "").strip().upper()
        tk = str(ldr.get("ticker") or "").strip().upper()
        tk_base = _IN_SUFFIX_STRIP_RE.sub("", tk.replace(".NS", "").replace(".BO", ""))
        return (raw_t in norm_uni) or (tk in norm_uni) or (tk_base in norm_uni)

    raw_alpha = leaders_raw[0]
    raw_dict_list = [ldr for ldr in leaders_raw if isinstance(ldr, dict)]
    scoped_pool = [ldr for ldr in raw_dict_list if _in_universe(ldr)]

    strict_rs_gate = min_rs is not None and float(min_rs) > 0

    def _is_clean_leader(ldr: dict, strict_adtv: bool = True, strict_bars: bool = True) -> bool:
        if strict_rs_gate:
            if float(ldr.get("rs", 0) or 0) < float(min_rs):
                return False
        if strict_bars:
            if ldr.get("is_short_history", False):
                return False
            bars = ldr.get("bars_count", 63)
            if bars is not None and int(bars) < min_bars:
                return False
        if strict_adtv:
            if not ldr.get("is_tradable_leader", True):
                return False
            if min_adtv is not None:
                adtv_val = float(ldr.get("adtv_cr", 0.0) or 0.0)
                if adtv_val < float(min_adtv):
                    return False
        return True

    def _select_from_pool(pool: list) -> list:
        picked = [ldr for ldr in pool if _is_clean_leader(ldr, strict_adtv=True, strict_bars=True)]
        if picked:
            picked_rs70 = [ldr for ldr in picked if float(ldr.get("rs", 0) or 0) >= 70.0]
            if picked_rs70:
                return picked_rs70[:max_n]
            if strict_rs_gate:
                return []
            return picked[:1]

        if strict_rs_gate or min_adtv is not None:
            return []

        picked_trad = [ldr for ldr in pool if _is_clean_leader(ldr, strict_adtv=True, strict_bars=False)]
        if picked_trad:
            picked_trad_rs70 = [ldr for ldr in picked_trad if float(ldr.get("rs", 0) or 0) >= 70.0]
            if picked_trad_rs70:
                return picked_trad_rs70[:max_n]
            return picked_trad[:1]

        pool_rs70 = [ldr for ldr in pool if float(ldr.get("rs", 0) or 0) >= 70.0]
        if pool_rs70:
            return pool_rs70[:max_n]
        return pool[:1]

    selected = _select_from_pool(scoped_pool)
    is_universe_fallback = False
    if not selected and norm_uni is not None and fallback_to_exchange and raw_dict_list:
        selected = _select_from_pool(raw_dict_list)
        if selected:
            is_universe_fallback = True

    tradable = []
    for pos_idx, ldr in enumerate(selected[:max_n]):
        item = dict(ldr)
        item["tradable_rank"] = pos_idx + 1
        item["is_universe_fallback"] = is_universe_fallback
        tradable.append(item)

    alpha = tradable[0] if (len(tradable) > 0 and int(tradable[0].get("rs", 0) or 0) >= 70) else None
    sister = tradable[1] if (len(tradable) > 1 and int(tradable[1].get("rs", 0) or 0) >= 70) else None
    leader_3 = tradable[2] if (len(tradable) > 2 and int(tradable[2].get("rs", 0) or 0) >= 70) else None

    raw_alpha_anomaly = bool(
        raw_alpha
        and alpha
        and str(raw_alpha.get("ticker", "")).upper() != str(alpha.get("ticker", "")).upper()
        and (raw_alpha.get("is_short_history", False) or not raw_alpha.get("is_tradable_leader", True))
    )

    return {
        "tradable_leaders": tradable,
        "alpha": alpha,
        "sister": sister,
        "leader_3": leader_3,
        "raw_alpha": raw_alpha,
        "raw_alpha_anomaly": raw_alpha_anomaly,
        "is_universe_fallback": is_universe_fallback,
    }


def _write_atomic_json(target_path: str, payload) -> None:
    """Atomically write JSON payload to disk using PID-scoped temp file."""
    tmp_path = target_path + f".tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp_path, target_path)


def is_industry_cache_stale(universe_scope: str = "India (NSE/BSE)") -> bool:
    """
    Checks if the on-disk industry intelligence snapshot is missing, mismatched in schema version,
    or older than historical_prices_matrix.pkl.
    """
    is_in = _scope_is_india(universe_scope)
    cache_path = CACHE_INTEL_INDIA_PATH if is_in else CACHE_INTEL_US_PATH
    matrix_path = CACHE_MATRIX_INDIA_PATH if is_in else CACHE_MATRIX_US_PATH
    niche_path = CACHE_NICHE_INDIA_PATH if is_in else CACHE_NICHE_US_PATH
    if not os.path.exists(cache_path) or not os.path.exists(matrix_path) or not os.path.exists(niche_path):
        return True
    try:
        sig = _get_file_signature(matrix_path)
        records = _load_json_payload_cached(matrix_path, sig)
        if not isinstance(records, list) or not records:
            return True
        if int(records[0].get("Cache_Schema_Version", 0) or 0) != CACHE_SCHEMA_VERSION:
            return True
    except Exception:
        return True

    _db_candidates = []
    try:
        from database import DATABASE_PATH
        if DATABASE_PATH:
            _db_candidates.append(DATABASE_PATH)
    except Exception:
        pass
    src_paths = [p for p in (MATRIX_PKL_PATH, MAP_JSON_PATH, *_db_candidates) if os.path.exists(p)]
    if not src_paths:
        return False
    try:
        cache_mtime = min(os.path.getmtime(cache_path), os.path.getmtime(matrix_path), os.path.getmtime(niche_path))
        latest_src_mtime = max(os.path.getmtime(p) for p in src_paths)
        return cache_mtime < latest_src_mtime
    except Exception:
        return True


def refresh_industry_group_intelligence_cache(
    universe_scope: str = "India (NSE/BSE)",
    force: bool = True,
    skip_enrichment: bool = None,
) -> dict:
    """
    Headless export function callable by EOD sync, app startup, or background scripts.
    1. Resolves unmapped tickers via auto-enrichment (unless skip_enrichment=True or force=False).
    2. Recomputes MarketSurge 2.0 industry group momentum matrix + niche table.
    3. Atomically saves .cache snapshots (intel, matrix, niche) to disk.
    4. Returns stock intelligence dictionary.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    is_in = _scope_is_india(universe_scope)
    cache_path = CACHE_INTEL_INDIA_PATH if is_in else CACHE_INTEL_US_PATH
    matrix_cache_path = CACHE_MATRIX_INDIA_PATH if is_in else CACHE_MATRIX_US_PATH
    niche_cache_path = CACHE_NICHE_INDIA_PATH if is_in else CACHE_NICHE_US_PATH
    if skip_enrichment is None:
        skip_enrichment = not bool(force)

    # Step 1: Auto-enrich missing tickers (skipped during fast local rebuilds / force=False)
    if not skip_enrichment:
        try:
            enriched_count = auto_enrich_missing_industries(universe_scope=universe_scope)
            if enriched_count > 0:
                print(f"[IndustryGroupEngine] Auto-enriched {enriched_count} missing tickers for {universe_scope}")
        except Exception as e:
            print(f"[IndustryGroupEngine] Auto-enrichment notice: {e}")

    # Step 2: Recompute matrix + niche table
    df_matrix, _, df_niche = compute_industry_group_matrix_with_niche(
        taxonomy="canonical", universe_scope=universe_scope
    )
    if df_matrix is None or df_matrix.empty:
        return {}

    # Step 3: Build intelligence payload (includes ranked & niche NR lookup)
    stock_intel = _build_stock_intelligence_dict(df_matrix, df_niche=df_niche)

    # Step 4: Atomic file persistence
    try:
        _write_atomic_json(cache_path, stock_intel)
        _write_atomic_json(matrix_cache_path, df_matrix.to_dict(orient="records"))
        niche_records = df_niche.to_dict(orient="records") if (df_niche is not None and not df_niche.empty) else []
        _write_atomic_json(niche_cache_path, niche_records)
    except Exception as e:
        print(f"[IndustryGroupEngine] Error saving persistent cache: {e}")

    return stock_intel


def load_cached_industry_group_matrix(universe_scope: str = "India (NSE/BSE)") -> pd.DataFrame:
    """
    Non-blocking mtime-keyed loader for persisted `.cache/industry_group_matrix_*.json` (<10ms).
    Hydrates from Turso on empty Cloud containers; returns empty DataFrame if absent everywhere.
    """
    _hydrate_igm_cache_from_turso_if_missing(universe_scope)
    is_in = _scope_is_india(universe_scope)
    matrix_cache_path = CACHE_MATRIX_INDIA_PATH if is_in else CACHE_MATRIX_US_PATH
    sig = _get_file_signature(matrix_cache_path)
    records = _load_json_payload_cached(matrix_cache_path, sig)
    if isinstance(records, list) and records:
        try:
            return pd.DataFrame(records)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def get_stock_group_intelligence(taxonomy: str = "canonical", universe_scope: str = "India (NSE/BSE)") -> dict:
    """
    Returns a unified cross-application intelligence dictionary mapping every ticker to its
    canonical Industry Group metadata, IBD Letter Grade (A+ to E), and Livermore Constituent Role
    (Alpha Leader #1, Sister Stock #2, Pack Leader, or Sympathy Laggard).

    Strictly non-blocking: reads on-disk JSON snapshots keyed by (st_mtime_ns, st_size) in <10ms
    without auto-triggering Yahoo Finance enrichment on stale caches. Hydrates from Turso on empty
    Cloud containers, or performs a fast no-network local build if the local price matrix exists.
    """
    # Defensive guard if caller passed universe_scope as the first positional argument
    if str(taxonomy).startswith(("India", "US", "IN")):
        universe_scope = taxonomy
        taxonomy = "canonical"

    _hydrate_igm_cache_from_turso_if_missing(universe_scope)
    cache_path = CACHE_INTEL_INDIA_PATH if _scope_is_india(universe_scope) else CACHE_INTEL_US_PATH
    sig = _get_file_signature(cache_path)

    # Fast path: Pure mtime-keyed persistent JSON snapshot (< 10ms)
    if sig != (0, 0):
        data = _load_json_payload_cached(cache_path, sig)
        if data and isinstance(data, dict) and len(data) > 50:
            return data

    # Bootstrap without network I/O ONLY if JSON snapshot doesn't exist yet on local workspace
    if not os.path.exists(cache_path) and os.path.exists(MATRIX_PKL_PATH):
        return refresh_industry_group_intelligence_cache(
            universe_scope=universe_scope, force=False, skip_enrichment=True
        )
    return {}

def get_top_and_worst_40_groups(df_matrix: pd.DataFrame) -> tuple:
    """
    Extracts non-overlapping Leading vs Lagging Industry Groups using O'Neil / IBD Top 40 vs Worst 40 slices:
    n_slice = min(40, m // 2).
    """
    if df_matrix is None or df_matrix.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_sorted = df_matrix.sort_values('Rank_Today', ascending=True).copy()
    m = len(df_sorted)
    if m < 2:
        return df_sorted.copy(), df_sorted.iloc[0:0].copy()

    n_slice = min(40, m // 2)
    top_slice = df_sorted.head(n_slice).copy()
    worst_slice = df_sorted.tail(n_slice).sort_values('Rank_Today', ascending=True).copy()
    return top_slice, worst_slice


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_cloud_group_ohlcv(tickers_tuple: tuple[str, ...]) -> tuple:
    """Fetch 1Y OHLCV panels for <=25 tickers of a single Industry Group on Cloud without .pkl."""
    tickers = [str(t).strip() for t in (tickers_tuple or ()) if t][:25]
    empty = pd.DataFrame()
    if not tickers:
        return empty, empty, empty, empty
    try:
        import yfinance as yf
        raw = yf.download(tickers, period="1y", auto_adjust=False, progress=False, threads=False)
        if raw is None or raw.empty:
            return empty, empty, empty, empty
        if isinstance(raw.columns, pd.MultiIndex):
            lvl0 = set(raw.columns.get_level_values(0))
            close_df = raw["Close"].copy() if "Close" in lvl0 else empty
            high_df = raw["High"].copy() if "High" in lvl0 else close_df.copy()
            low_df = raw["Low"].copy() if "Low" in lvl0 else close_df.copy()
            vol_df = raw["Volume"].copy() if "Volume" in lvl0 else pd.DataFrame(0.0, index=close_df.index, columns=close_df.columns)
        else:
            t0 = tickers[0]
            close_df = pd.DataFrame({t0: raw["Close"]}) if "Close" in raw.columns else empty
            high_df = pd.DataFrame({t0: raw["High"]}) if "High" in raw.columns else close_df.copy()
            low_df = pd.DataFrame({t0: raw["Low"]}) if "Low" in raw.columns else close_df.copy()
            vol_df = pd.DataFrame({t0: raw["Volume"]}) if "Volume" in raw.columns else pd.DataFrame(0.0, index=close_df.index, columns=close_df.columns)
        for df_p in (close_df, high_df, low_df, vol_df):
            if df_p is not None and not df_p.empty:
                df_p.index = pd.to_datetime(df_p.index).tz_localize(None).normalize()
        return close_df, high_df, low_df, vol_df
    except Exception:
        return empty, empty, empty, empty


def _collect_group_tickers_for_cloud_deep_dive(
    group_name: str,
    taxonomy: str,
    universe_scope: str,
) -> tuple[list[str], dict[str, int]]:
    """Collect <=25 constituent tickers and cached universal RS ratings from IGM snapshots."""
    tickers: list[str] = []
    cached_rs_map: dict[str, int] = {}
    is_in = _scope_is_india(universe_scope)
    df_candidates = []
    if str(taxonomy).lower().strip() == "thematic":
        df_t, _, _ = _load_turso_igm_bundle("thematic", universe_scope)
        if not df_t.empty:
            df_candidates.append(df_t)
    df_m = load_cached_industry_group_matrix(universe_scope)
    df_n = load_cached_niche_industry_groups(universe_scope)
    for df_c in (df_m, df_n):
        if df_c is not None and not df_c.empty:
            df_candidates.append(df_c)

    for df_c in df_candidates:
        if "Industry_Group" not in df_c.columns:
            continue
        sub = df_c[df_c["Industry_Group"] == group_name]
        if sub.empty:
            continue
        row = sub.iloc[0]
        for ldr in (row.get("Top_Leaders_Full") or []):
            if not isinstance(ldr, dict):
                continue
            rt = str(ldr.get("raw_ticker") or "").strip()
            tk = str(ldr.get("ticker") or "").strip()
            sym = rt or (tk if (not is_in or tk.endswith((".NS", ".BO"))) else f"{tk}.NS")
            if sym and sym not in tickers:
                tickers.append(sym)
            rs_v = ldr.get("rs")
            if rs_v is not None and sym:
                cached_rs_map[sym] = int(float(rs_v))
                if tk:
                    cached_rs_map[tk] = int(float(rs_v))
        for tk in (row.get("Constituents") or []):
            sym = str(tk).strip()
            if sym and (not is_in or sym.endswith((".NS", ".BO"))) and sym not in tickers:
                tickers.append(sym)

    group_map = _resolve_group_map(taxonomy=taxonomy, universe_scope=universe_scope)
    for sym in (group_map.get(group_name, []) or []):
        s_str = str(sym).strip()
        if s_str and s_str not in tickers:
            tickers.append(s_str)

    return tickers[:25], cached_rs_map


def get_group_deep_dive_data(
    group_name: str,
    taxonomy: str = "canonical",
    universe_scope: str = "India (NSE/BSE)",
    weight_mode: str = "capped_mcap",
):
    """
    Extracts deep-dive constituent table and MarketSurge 2.0 capped market-cap vs benchmark curve.
    Reuses shared universe hygiene, smoothed 6M RS, share-class dedup, ADTV tiers, and RSNH/Blue-Dot detectors.
    """
    close_df, high_df, low_df, vol_df, core_bench_set = _prepare_universe_price_panels(universe_scope)
    cached_rs_overlay: dict[str, int] = {}
    if close_df.empty:
        cloud_tickers, cached_rs_overlay = _collect_group_tickers_for_cloud_deep_dive(
            group_name=group_name,
            taxonomy=taxonomy,
            universe_scope=universe_scope,
        )
        if not cloud_tickers:
            return pd.DataFrame(), pd.DataFrame(), ""
        close_df, high_df, low_df, vol_df = _fetch_cloud_group_ohlcv(tuple(cloud_tickers))
        if close_df.empty:
            return pd.DataFrame(), pd.DataFrame(), ""
        core_bench_set = set()

    is_us = not _scope_is_india(universe_scope)
    close_clean = close_df.ffill(limit=5)
    n_bars = len(close_clean)
    end_idx = n_bars - 1
    c0_all = close_clean.iloc[-1]

    mcap_map, _, name_map = _load_db_enrichment_maps()
    tradable_universe_set, stock_adtv_map, _ = _compute_capped_adtv_and_tradables(
        close_df=close_df,
        high_df=high_df,
        low_df=low_df,
        volume_df=vol_df,
        core_benchmark_set=core_bench_set,
        mcap_map=mcap_map,
        is_india=(not is_us),
    )
    tradable_cols_list = [
        c for c in close_clean.columns
        if c in tradable_universe_set and pd.notna(c0_all.get(c)) and float(c0_all.get(c) or 0.0) > 0
    ]
    if not tradable_cols_list:
        tradable_cols_list = [c for c in close_clean.columns if pd.notna(c0_all.get(c)) and float(c0_all.get(c) or 0.0) > 0]

    rs_pct_all, comp_all, _ = _compute_smoothed_6m_rs_at_offset(close_clean, set(tradable_cols_list), end_idx)
    if cached_rs_overlay:
        for t_k, r_val in cached_rs_overlay.items():
            rs_pct_all[t_k] = r_val
    bench_series, _ = load_benchmark_close(universe_scope)
    is_rsnh_s, is_blue_dot_s, _ = _compute_rsnh_panels(close_clean, tradable_cols_list, bench_series=bench_series)

    group_map = _resolve_group_map(taxonomy=taxonomy, universe_scope=universe_scope)
    tickers = list(group_map.get(group_name, []) or [])
    for c_col in close_clean.columns:
        if c_col not in tickers and c_col in cached_rs_overlay:
            tickers.append(c_col)
    avail_raw = [t for t in tickers if t in close_clean.columns and pd.notna(c0_all.get(t)) and float(c0_all.get(t) or 0.0) > 0]
    if not avail_raw and cached_rs_overlay:
        avail_raw = [t for t in close_clean.columns if pd.notna(c0_all.get(t)) and float(c0_all.get(t) or 0.0) > 0]
    if not avail_raw:
        return pd.DataFrame(), pd.DataFrame(), ""

    dedup_status = {
        t: {
            'adtv': float(stock_adtv_map.get(t, 0.0) or 0.0),
            'rs': int(rs_pct_all.get(t, 0)),
            'is_tradable': t in tradable_universe_set,
        }
        for t in avail_raw
    }
    avail, _ = _dedupe_share_classes(avail_raw, dedup_status, mcap_map=mcap_map, company_name_map=name_map)
    if not avail:
        return pd.DataFrame(), pd.DataFrame(), ""

    sub_close = close_clean[avail]
    sub_high = high_df[[t for t in avail if t in high_df.columns]]
    if sub_high.empty:
        sub_high = sub_close

    c0 = sub_close.ffill().iloc[-1]
    c_1d = sub_close.ffill().iloc[-2] if n_bars > 1 else c0
    c_1w = sub_close.ffill().iloc[-6] if n_bars > 5 else c0
    c_1m = sub_close.ffill().iloc[-22] if n_bars > 21 else c0
    c_3m = sub_close.ffill().iloc[-64] if n_bars > 63 else c0
    c_1y = sub_close.ffill().iloc[max(0, n_bars - 250)]

    pivot_25 = sub_high.iloc[-26:-1].max() if len(sub_high) >= 27 else sub_high.max()
    ema_21 = sub_close.ffill().ewm(span=21, adjust=False).mean().iloc[-1]
    sma_50_dd = sub_close.ffill().rolling(50, min_periods=20).mean().iloc[-1]
    sma_200_dd = sub_close.ffill().rolling(200, min_periods=50).mean().iloc[-1]

    r1d = ((c0 / c_1d - 1.0) * 100.0).round(2)
    r1w = ((c0 / c_1w - 1.0) * 100.0).round(2)
    r1m = ((c0 / c_1m - 1.0) * 100.0).round(2)
    r3m = ((c0 / c_3m - 1.0) * 100.0).round(2)
    r1y = ((c0 / c_1y - 1.0) * 100.0).round(2)

    dist_piv = (((c0 - pivot_25) / pivot_25.replace(0.0, np.nan)) * 100.0).fillna(0.0).round(2)
    dist_ema = (((c0 - ema_21) / ema_21.replace(0.0, np.nan)) * 100.0).fillna(0.0).round(2)

    mar_rate = 0.045 if is_us else 0.065
    cmp_label = 'CMP ($)' if is_us else 'CMP (₹)'

    constit_rows = []
    tv_tickers = []

    for t in avail:
        clean_t = t.replace('.NS', '').replace('.BO', '')
        if is_us:
            tv_tickers.append(clean_t)
            tv_url = f"https://www.tradingview.com/chart/?symbol={clean_t}"
        else:
            exchange = 'BSE' if t.endswith('.BO') else 'NSE'
            tv_tickers.append(f"{exchange}:{clean_t}")
            tv_url = f"https://www.tradingview.com/chart/?symbol={exchange}%3A{clean_t}"

        p_dist = float(dist_piv.get(t, 0.0))
        e_dist = float(dist_ema.get(t, 0.0))
        c_val_t = float(c0.get(t, np.nan))
        m50_t = float(sma_50_dd.get(t, np.nan))
        m200_t = float(sma_200_dd.get(t, np.nan))
        abv_50_t = True if (np.isnan(c_val_t) or np.isnan(m50_t)) else bool(c_val_t >= m50_t)
        abv_200_t = True if (np.isnan(c_val_t) or np.isnan(m200_t)) else bool(c_val_t >= m200_t)
        status_lbl, badge_style, prio = classify_stock_action_state(
            p_dist, e_dist, above_50sma=abv_50_t, above_200sma=abv_200_t
        )

        c_series_t = sub_close[t].dropna()
        bars_cnt = len(c_series_t)
        s3_t = calculate_sortino_ratio(c_series_t, window=63, mar_annual=mar_rate) if bars_cnt >= 63 else 0.0
        s6_t = calculate_sortino_ratio(c_series_t, window=126, mar_annual=mar_rate) if bars_cnt >= 126 else 0.0
        h_s_t = high_df[t].dropna() if t in high_df.columns else None
        l_s_t = low_df[t].dropna() if (low_df is not None and t in low_df.columns) else None
        sm_t = calculate_smooth_rs_atr(c_series_t, h_s_t, l_s_t) if bars_cnt >= 30 else 0.0

        adtv_val = float(stock_adtv_map.get(t, 0.0) or 0.0)
        is_tradable_t = t in tradable_universe_set
        is_bench_t = (t in core_bench_set or clean_t in core_bench_set or f"{clean_t}.NS" in core_bench_set)
        tier_lbl = "Core" if is_bench_t else ("Tradable" if is_tradable_t else "🚫 Tier-0 Illiquid")
        if is_us:
            adtv_str = f"${adtv_val / 1e6:.1f}M" if adtv_val >= 1e6 else f"${adtv_val / 1e3:.0f}K"
        else:
            adtv_str = f"₹{adtv_val / 1e7:.1f}Cr" if adtv_val >= 1e7 else f"₹{adtv_val / 1e5:.0f}L"

        blue_t = bool(is_blue_dot_s.get(t, False))
        rsnh_t = bool(is_rsnh_s.get(t, False))
        rs_line_lbl = "🎯 Blue Dot" if blue_t else ("🔵 RS NH" if rsnh_t else "—")

        constit_rows.append({
            'Symbol': clean_t,
            'Raw_Ticker': t,
            'CMP': float(c0.get(t, 0.0)),
            'CMP (₹)': float(c0.get(t, 0.0)),
            cmp_label: float(c0.get(t, 0.0)),
            '1D %': float(r1d.get(t, 0.0)),
            '1W %': float(r1w.get(t, 0.0)),
            '1M %': float(r1m.get(t, 0.0)),
            '3M %': float(r3m.get(t, 0.0)),
            '1Y %': float(r1y.get(t, 0.0)),
            'RS Rating': int(rs_pct_all.get(t, 0)),
            'Sortino 3M': round(s3_t, 2),
            'Sortino 6M': round(s6_t, 2),
            'Smooth RS': round(sm_t, 2),
            'Pivot Price': round(float(pivot_25.get(t, 0.0)), 2),
            'Dist Pivot %': p_dist,
            'Dist 21 EMA %': e_dist,
            'Execution Status': status_lbl,
            'Status_Style': badge_style,
            'Status_Priority': prio,
            'Tier / Liquidity': tier_lbl,
            'ADTV': adtv_str,
            'RS Line High': rs_line_lbl,
            '_Is_Tradable': is_tradable_t,
            '_Is_IPO': bars_cnt < 63,
            'TradingView_URL': tv_url,
        })

    df_constits = (
        pd.DataFrame(constit_rows)
        .sort_values(['_Is_Tradable', '_Is_IPO', 'Status_Priority', 'RS Rating'], ascending=[False, True, True, False])
        .drop(columns=['_Is_Tradable', '_Is_IPO'])
        .reset_index(drop=True)
    )

    window_bars = min(250, len(sub_close))
    sub_window = sub_close.iloc[-window_bars:]
    curve_tickers = [t for t in avail if t in tradable_universe_set] or avail
    norm_group, _ = _build_group_curve(sub_window, curve_tickers, mcap_map=mcap_map, weight_mode=weight_mode)
    norm_group = norm_group.reindex(sub_window.index).ffill().bfill()

    if bench_series is not None and not bench_series.empty:
        b_win = bench_series.reindex(sub_window.index, method="ffill").bfill()
        first_b = float(b_win.iloc[0]) if not b_win.empty and pd.notna(b_win.iloc[0]) and float(b_win.iloc[0]) > 0 else 1.0
        norm_bench = (b_win / first_b) * 100.0
    else:
        bench_sub = close_clean[tradable_cols_list].iloc[-window_bars:]
        bench_rets = bench_sub.pct_change(fill_method=None).mean(axis=1, skipna=True).fillna(0.0)
        norm_bench = (1.0 + bench_rets).cumprod() * 100.0

    rs_line_curve = (norm_group / norm_bench.replace(0.0, np.nan)) * 100.0
    curve_df = pd.DataFrame({
        'Industry Group': norm_group.round(2),
        'Universe Benchmark': norm_bench.round(2),
        'Group RS Line': rs_line_curve.round(2),
    })

    tv_copy_box = ",".join(tv_tickers)
    return df_constits, curve_df, tv_copy_box


# =============================================================================
# MASTER TOP-40 TRADINGVIEW BATCH WATCHLIST EXPORTER HELPERS
# =============================================================================

_IN_SUFFIX_STRIP_RE = re.compile(r"-(?:SM|ST|SZ|BE|BZ)$", re.IGNORECASE)
_SETUP_PRIO_KEYWORDS = (
    ("IN BUY ZONE", 0),
    ("BUY", 0),
    ("RETEST", 1),
    ("COIL", 2),
    ("BASE", 3),
    ("EXT", 4),
    ("FAIL", 5),
)


def _setup_priority_rank(status: str) -> int:
    s_up = str(status or "").upper()
    for kw, prio in _SETUP_PRIO_KEYWORDS:
        if kw in s_up:
            return prio
    return 3


def format_tradingview_symbol(
    ticker: str,
    raw_ticker: str = "",
    is_us: bool = False,
    us_tv_map: dict = None,
    prefix_us: bool = True,
) -> str:
    """
    Format an internal ticker into a TradingView-exact watchlist symbol:
      - India: strips Yahoo suffixes (.NS/.BO/-SM/-ST/-SZ/-BE/-BZ), normalizes '-'/'&' -> '_',
               routes BSE-only symbols to BSE: and NSE listings to NSE:.
      - US: normalizes class dots (BRK-B -> BRK.B), optionally prepending NASDAQ:/NYSE:/AMEX:
            from security_master.tv_ticker to prevent Crypto/TSX ticker collisions.
    """
    raw_up = str(raw_ticker or ticker or "").strip().upper()
    tk_up = str(ticker or raw_ticker or "").strip().upper()

    if is_us:
        base = tk_up.replace(".NS", "").replace(".BO", "")
        tv_dot = base.replace("-", ".")
        if prefix_us and us_tv_map:
            mapped = (
                us_tv_map.get(raw_up)
                or us_tv_map.get(base)
                or us_tv_map.get(tv_dot)
            )
            if mapped and ":" in str(mapped):
                exch, sym_part = str(mapped).split(":", 1)
                return f"{exch.strip().upper()}:{sym_part.strip().upper().replace('-', '.')}"
        return tv_dot

    is_bse = raw_up.endswith(".BO")
    clean = raw_up.replace(".NS", "").replace(".BO", "") if raw_up.endswith((".NS", ".BO")) else tk_up
    clean = _IN_SUFFIX_STRIP_RE.sub("", clean)
    tv_sym = clean.replace("-", "_").replace("&", "_")
    exch = "BSE" if is_bse else "NSE"
    return f"{exch}:{tv_sym}"


def load_exporter_enrichment_maps(is_us: bool = False) -> tuple:
    """
    Batch-load (us_tv_map, funda_map) once per render from Turso/SQLite via database.py.
    Returns ({}, {}) safely if DB/tables are unavailable.
    """
    us_tv_map = {}
    funda_map = {}
    try:
        from database import get_fundamentals_map_for_engine, read_sql_df

        if is_us:
            try:
                df_sm = read_sql_df(
                    "SELECT symbol, yf_ticker, tv_ticker FROM security_master WHERE market = 'US' AND status = 'ACTIVE'"
                )
                if df_sm is not None and not df_sm.empty:
                    for row_d in df_sm.to_dict(orient="records"):
                        tv_t = row_d.get("tv_ticker")
                        yf_t = row_d.get("yf_ticker")
                        sym = row_d.get("symbol")
                        if tv_t and ":" in str(tv_t):
                            if yf_t:
                                us_tv_map[str(yf_t).strip().upper()] = str(tv_t).strip().upper()
                            if sym:
                                us_tv_map[str(sym).strip().upper()] = str(tv_t).strip().upper()
            except Exception:
                pass

        raw_funda, _ = get_fundamentals_map_for_engine("ALL")
        for tk_u, info in (raw_funda or {}).items():
            if not isinstance(info, dict):
                continue
            payload = {
                "eps_growth": float(info.get("eps_growth", 0.0) or 0.0),
                "sales_growth": float(info.get("sales_growth", 0.0) or 0.0),
                "roe": float(info.get("roe", 0.0) or 0.0),
                "market_cap": float(info.get("market_cap", 0.0) or 0.0),
            }
            funda_map[tk_u] = payload
            clean_k = _IN_SUFFIX_STRIP_RE.sub("", tk_u.replace(".NS", "").replace(".BO", ""))
            funda_map.setdefault(clean_k, payload)
    except Exception:
        pass

    return us_tv_map, funda_map


def build_top_n_tv_watchlist(
    df_source: pd.DataFrame,
    rank_range: tuple = (1, 40),
    grade_filter: str = "All in Rank Range",
    exclude_solitary: bool = True,
    dual_alpha_only: bool = False,
    allowed_groups: set = None,
    leadership_mode: str = "Top 3 Leaders Only",
    backfill_top_k: bool = False,
    wolf_pack_floor: str = "Same as Min Stock RS",
    max_pack_extras: int = 5,
    custom_top_k: int = 5,
    min_rs: int = 88,
    min_sortino: float = 1.5,
    exempt_short_history_ipo: bool = True,
    min_smooth_rs: float = -5.0,
    min_adtv: float = 0.0,
    exclude_circuit_locked: bool = True,
    hide_failed: bool = True,
    setup_scope: str = "All Non-Failed (Incl. Extended)",
    max_dist_ema: float = None,
    can_growth_gate: str = "All (No Fundamental Gate)",
    sort_export_by: str = "Hierarchy Rank (#1 to #N)",
    is_us: bool = False,
    prefix_us_exchanges: bool = True,
    us_tv_map: dict = None,
    funda_map: dict = None,
) -> dict:
    """
    CANSLIM 2-stage batch TradingView watchlist builder:
      Stage 1 (Group Gate): Strict Rank_Today slice + solitary/grade/dual-alpha/global guards.
      Stage 2 (Tradable Leadership): Liquidity + circuit-lock hygiene -> Top 3 / +Wolf Pack / Apex slice.
      Stage 3 (Quality Qualification): RS, Sortino 3M (with <63d IPO exemption), Smooth RS, Setup & CANSLIM EPS/Sales gates.
    """
    empty_res = {
        "rows": [],
        "groups_matched": 0,
        "groups_scanned": 0,
        "flat_tickers": [],
        "flat_csv": "",
        "sectioned_paste": "",
        "sectioned_txt": "",
        "unique_count": 0,
        "section_entry_count": 0,
        "apex_count": 0,
        "ipo_exempt_count": 0,
    }
    if df_source is None or df_source.empty or "Rank_Today" not in df_source.columns:
        return empty_res

    r_min, r_max = int(rank_range[0]), int(rank_range[1])
    grp_df = df_source.sort_values("Rank_Today", ascending=True).copy()
    grp_df = grp_df[(grp_df["Rank_Today"] >= r_min) & (grp_df["Rank_Today"] <= r_max)]

    if allowed_groups is not None:
        grp_df = grp_df[grp_df["Industry_Group"].isin(allowed_groups)]
    if exclude_solitary:
        cnt_col = "Tradable_Count" if "Tradable_Count" in grp_df.columns else ("Stock_Count" if "Stock_Count" in grp_df.columns else None)
        if cnt_col is not None:
            grp_df = grp_df[grp_df[cnt_col] >= MIN_RANKED_GROUP_SIZE_CANONICAL]
    if grade_filter == "A+ / A / B Only" and "Group_Grade" in grp_df.columns:
        grp_df = grp_df[grp_df["Group_Grade"].isin(["A+", "A", "B"])]
    elif grade_filter == "A+ & A Only (Top 20%)" and "Group_Grade" in grp_df.columns:
        grp_df = grp_df[grp_df["Group_Grade"].isin(["A+", "A"])]
    if dual_alpha_only and "Is_Dual_Alpha" in grp_df.columns:
        grp_df = grp_df[grp_df["Is_Dual_Alpha"] == True]

    if grp_df.empty:
        return empty_res

    f_map = funda_map or {}
    tv_map = us_tv_map or {}
    groups_scanned = len(grp_df)

    def _lookup_funda(tk: str, raw_tk: str) -> dict:
        r_u = str(raw_tk or "").strip().upper()
        t_u = str(tk or "").strip().upper()
        return f_map.get(r_u) or f_map.get(t_u) or {}

    def _passes_quality(c: dict, rs_floor: int, apex_mode: bool = False) -> bool:
        rs_val = int(c.get("rs", 0))
        if rs_val < rs_floor:
            return False

        bars_cnt = int(c.get("bars_count", 63))
        is_short_ipo = bool(c.get("is_short_history", bars_cnt < 63))
        s3 = float(c.get("sortino_3m", 0.0))
        sm_rs = float(c.get("smooth_rs", 0.0))

        sort_req = max(3.0, float(min_sortino)) if apex_mode else float(min_sortino)
        if not (exempt_short_history_ipo and is_short_ipo):
            if s3 < sort_req:
                return False
            if min_smooth_rs > -5.0 and bars_cnt >= 30 and sm_rs < float(min_smooth_rs):
                return False

        status = str(c.get("status", ""))
        prio = _setup_priority_rank(status)
        if hide_failed and prio == 5:
            return False
        if "Buyable Bases" in setup_scope and prio not in (0, 1, 2, 3):
            return False
        if "Immediate Buy" in setup_scope and prio not in (0, 1):
            return False

        if max_dist_ema is not None:
            dist_e = float(c.get("dist_ema", 0.0))
            if dist_e > float(max_dist_ema):
                return False

        if can_growth_gate != "All (No Fundamental Gate)":
            fn = _lookup_funda(c.get("ticker", ""), c.get("raw_ticker", ""))
            eps_g = float(fn.get("eps_growth", 0.0))
            sls_g = float(fn.get("sales_growth", 0.0))
            if "Positive EPS" in can_growth_gate and eps_g <= 0.0:
                return False
            if "CANSLIM Growth" in can_growth_gate and not (eps_g >= 25.0 or sls_g >= 25.0):
                return False

        return True

    exported_rows = []
    section_chunks = []
    groups_matched = 0
    ipo_exempt_count = 0
    apex_exported_count = 0

    for _, g_row in grp_df.iterrows():
        constits = g_row.get("Top_Leaders_Full") or g_row.get("Top_3_Leaders") or []
        if not constits:
            continue

        # Stage 2A: Tradable liquidity + circuit hygiene
        tradable = []
        for c in constits:
            if float(c.get("adtv_cr", 0.0)) < float(min_adtv):
                continue
            if exclude_circuit_locked and int(c.get("circuit_days_20d", 0)) >= 4:
                continue
            tradable.append(c)

        if not tradable:
            continue

        # Stage 2B + 3: Cohort slice + constituent quality checks
        chosen = []
        if leadership_mode == "Top 3 Leaders Only":
            pool = [c for c in tradable if _passes_quality(c, min_rs)] if backfill_top_k else tradable[:3]
            chosen = [c for c in pool if _passes_quality(c, min_rs)][:3]
        elif leadership_mode == "Top 3 + Wolf Pack":
            pack_rs_floor = 80 if "80" in wolf_pack_floor else int(min_rs)
            eff_pack_floor = min(int(min_rs), 80) if "80" in wolf_pack_floor else int(min_rs)
            if backfill_top_k:
                top3_cand = [c for c in tradable if _passes_quality(c, min_rs)][:3]
                used = {c.get("raw_ticker") or c.get("ticker") for c in top3_cand}
                pack_cand = [
                    c for c in tradable
                    if (c.get("raw_ticker") or c.get("ticker")) not in used
                    and int(c.get("rs", 0)) >= pack_rs_floor
                    and _passes_quality(c, eff_pack_floor)
                ]
            else:
                top3_cand = [c for c in tradable[:3] if _passes_quality(c, min_rs)]
                pack_cand = [
                    c for c in tradable[3:]
                    if int(c.get("rs", 0)) >= pack_rs_floor
                    and _passes_quality(c, eff_pack_floor)
                ]
            if max_pack_extras is not None and max_pack_extras >= 0:
                pack_cand = pack_cand[: int(max_pack_extras)]
            chosen = top3_cand + pack_cand
        elif "Apex Leaders" in leadership_mode:
            eff_apex_rs = max(80, int(min_rs))
            chosen = [c for c in tradable if _passes_quality(c, eff_apex_rs, apex_mode=True)]
        elif "Custom Top K" in leadership_mode:
            k = max(1, int(custom_top_k))
            pool = [c for c in tradable if _passes_quality(c, min_rs)] if backfill_top_k else tradable[:k]
            chosen = [c for c in pool if _passes_quality(c, min_rs)][:k]
        else:  # All Qualifying Constituents
            chosen = [c for c in tradable if _passes_quality(c, min_rs)]

        if not chosen:
            continue

        groups_matched += 1
        ind_rank = int(g_row.get("Rank_Today", 0))
        ind_name = str(g_row.get("Industry_Group", ""))
        ind_grade = str(g_row.get("Group_Grade", ""))
        grp_s3m = float(g_row.get("Sortino_3M", 0.0))

        sec_symbols = []
        for c in chosen:
            tk = str(c.get("ticker", ""))
            raw_tk = str(c.get("raw_ticker", tk))
            tv_sym = format_tradingview_symbol(
                tk,
                raw_ticker=raw_tk,
                is_us=is_us,
                us_tv_map=tv_map,
                prefix_us=prefix_us_exchanges,
            )
            bars_cnt = int(c.get("bars_count", 63))
            is_short_ipo = bool(c.get("is_short_history", bars_cnt < 63))
            s3 = float(c.get("sortino_3m", 0.0))
            rs_val = int(c.get("rs", 0))
            fn = _lookup_funda(tk, raw_tk)

            if is_short_ipo:
                ipo_exempt_count += 1
            if rs_val >= 80 and s3 >= 3.0:
                apex_exported_count += 1

            sec_symbols.append(tv_sym)
            role_disp = f"{c.get('role_icon', '')} {c.get('role_title', '')}".strip()
            if is_short_ipo:
                role_disp = f"🆕 IPO({bars_cnt}D) · {role_disp}"

            exported_rows.append({
                "Ind #": ind_rank,
                "Industry": ind_name,
                "Grade": ind_grade,
                "Grp Rank/Idx": f"#{ind_rank}",
                "Stock Rank": int(c.get("rank", 0)),
                "Role": role_disp,
                "Symbol": tk,
                "TV Ticker": tv_sym,
                "RS": rs_val,
                "Smooth RS": round(float(c.get("smooth_rs", 0.0)), 2),
                "Sortino 3M": round(s3, 2),
                "ADTV": round(float(c.get("adtv_cr", 0.0)), 2),
                "EPS YoY %": round(float(fn.get("eps_growth", 0.0)), 1),
                "Sales YoY %": round(float(fn.get("sales_growth", 0.0)), 1),
                "Dist 21EMA %": round(float(c.get("dist_ema", 0.0)), 1),
                "Dist Pivot": str(c.get("dist_pivot_str", "")),
                "Setup Status": str(c.get("status", "")),
                "_setup_prio": _setup_priority_rank(c.get("status", "")),
            })

        section_chunks.append((f"### #{ind_rank} {ind_name} ({ind_grade} | GrpS {grp_s3m:.1f})", sec_symbols))

    if not exported_rows:
        return {**empty_res, "groups_scanned": groups_scanned}

    if "Actionable" in sort_export_by:
        exported_rows.sort(key=lambda r: (r["_setup_prio"], r["Ind #"], r["Stock Rank"]))
    else:
        exported_rows.sort(key=lambda r: (r["Ind #"], r["Stock Rank"]))

    for r in exported_rows:
        r.pop("_setup_prio", None)

    flat_tickers = list(dict.fromkeys(r["TV Ticker"] for r in exported_rows if r.get("TV Ticker")))
    flat_csv = ",".join(flat_tickers)

    sectioned_lines = []
    for hdr, syms in section_chunks:
        sectioned_lines.append(hdr)
        sectioned_lines.extend(syms)
    sectioned_txt = "\n".join(sectioned_lines)
    sectioned_paste = "\n".join(sectioned_lines)
    section_entry_count = sum(len(syms) for _, syms in section_chunks)

    return {
        "rows": exported_rows,
        "groups_matched": groups_matched,
        "groups_scanned": groups_scanned,
        "flat_tickers": flat_tickers,
        "flat_csv": flat_csv,
        "sectioned_paste": sectioned_paste,
        "sectioned_txt": sectioned_txt,
        "unique_count": len(flat_tickers),
        "section_entry_count": section_entry_count,
        "apex_count": apex_exported_count,
        "ipo_exempt_count": ipo_exempt_count,
    }
