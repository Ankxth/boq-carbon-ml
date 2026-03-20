"""
Part 1: Data Ingestion and Cleaning
=====================================
Project : BOQ Embodied Carbon Estimation using ML
File    : src/part1_ingestion.py

What this script does:
  1. Loads all 5 BOQ files (Bot, Eco, Mall, Zen, PA)
  2. Standardises columns: description, unit, quantity, rate (PA only)
  3. Propagates material identity from section headers to child rows
  4. Propagates concrete grade from grade-spec rows to child rows
  5. Cleans text for NLP (lowercases, strips punctuation, collapses whitespace)
  6. Standardises units (cum->m3, sqm->m2, rmt->m, etc.)
  7. Drops zero-quantity rows, junk totals, empty descriptions
  8. Deduplicates within each source file
  9. Saves: data/processed/master_cleaned.csv

Fixes applied vs previous version:
  - Electrical conduit no longer misclassified as steel
  - Tile/granite flooring no longer misclassified as sand
  - Formwork added as its own material category
  - Flooring added as its own material category
  - Section-break keywords tightened (formwork removed from list)
  - Concrete m2 rows: thickness extracted from description where stated
  - PA rates extracted and stored alongside BOQ items

Run:
  cd boq_carbon_ml
  python src/part1_ingestion.py
"""

import pandas as pd
import numpy as np
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE  = Path(__file__).resolve().parent.parent
RAW   = BASE / "data" / "raw"
PROC  = BASE / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — UNIT STANDARDISATION MAP
# Every variant seen across the 5 files mapped to a canonical form
# ─────────────────────────────────────────────────────────────────────────────

UNIT_MAP = {
    # Cubic metre
    "cum":"m3","m3":"m3","cmt":"m3","cu.m":"m3","cu m":"m3",
    "cubic meter":"m3","cubic metre":"m3","m³":"m3",
    # Square metre
    "sqm":"m2","m2":"m2","sq.m":"m2","sq m":"m2","sqmt":"m2",
    "sq.mt":"m2","smt":"m2","m²":"m2",
    # Linear metre
    "rmt":"m","rm":"m","lm":"m","running meter":"m",
    "rmt.":"m","r.m":"m","mtr":"m","mtr.":"m",
    # Weight
    "mt":"ton","mts":"ton","tonne":"ton","tonnes":"ton",
    "kg":"kg","kgs":"kg","kilo":"kg",
    # Count
    "nos":"nos","no":"nos","number":"nos","each":"nos",
    "no.":"nos","nos.":"nos","no's":"nos","nrs":"nos","nr":"nos",
    "unit":"nos","each":"nos",
    # Lump sum
    "ls":"ls","lump sum":"ls","lot":"ls","l.s":"ls","l.s.":"ls","job":"ls",
    # Square feet
    "sqft":"sqft","sft":"sqft","sq.ft":"sqft",
    # Day / bags (PA specific)
    "day":"day","bags":"bags",
}

def standardise_unit(raw) -> str:
    if pd.isna(raw):
        return "unknown"
    s = str(raw).lower().strip()
    return UNIT_MAP.get(s, s)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text) -> str:
    """Lowercase, collapse whitespace, remove punctuation. Keep numbers."""
    if pd.isna(text):
        return ""
    t = str(text).lower()
    t = re.sub(r'\s+', ' ', t).strip()
    t = re.sub(r'[^\w\s]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


def clean_text_nlp(text) -> str:
    """Further strip inline dimensions for NLP matching."""
    t = clean_text(text)
    t = re.sub(r'\b\d+\s*(mm|cm|mtr|thk|dia)\b', '', t)
    return re.sub(r'\s+', ' ', t).strip()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — GRADE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

CONCRETE_GRADE_PAT = re.compile(
    r'\b(m\s?5|m\s?10|m\s?15|m\s?20|m\s?25|m\s?30|m\s?35|m\s?40|m\s?45|m\s?50)\b',
    re.IGNORECASE
)

# Zen.xlsx uses mix ratios instead of M-grades
MIX_RATIO_MAP = [
    (re.compile(r'1\s*:\s*5\s*:\s*10'), 'M5'),
    (re.compile(r'1\s*:\s*4\s*:\s*8'),  'M10'),
    (re.compile(r'1\s*:\s*3\s*:\s*6'),  'M15'),
    (re.compile(r'1\s*:\s*2\s*:\s*4'),  'M20'),
]

STEEL_GRADE_PAT = re.compile(
    r'\b(fe\s?250|fe\s?415|fe\s?500|fe\s?500d|fe\s?550|fe\s?550d|fe\s?600)\b',
    re.IGNORECASE
)


def extract_concrete_grade(text: str):
    m = CONCRETE_GRADE_PAT.search(text)
    if m:
        return re.sub(r'\s', '', m.group(1)).upper()
    for pat, grade in MIX_RATIO_MAP:
        if pat.search(text):
            return grade
    return None


def extract_steel_grade(text: str):
    m = STEEL_GRADE_PAT.search(text)
    return re.sub(r'\s', '', m.group(1)).upper() if m else None


def extract_thickness_mm(text: str):
    """
    Extract stated thickness from description for m2 concrete rows.
    e.g. '75mm THK' -> 75, '50mm thick' -> 50, '100mm thickness' -> 100
    Returns float mm or None.
    """
    pat = re.compile(r'(\d+)\s*mm\s*(thk|thick|thickness)', re.IGNORECASE)
    m = pat.search(text)
    if m:
        return float(m.group(1))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — MATERIAL DETECTION PATTERNS
#
# ORDER MATTERS.
# Each material is tried in order; first match wins.
# Key fixes vs previous version:
#   - "pipe" patterns come BEFORE "steel" to catch conduit correctly
#   - "tile" patterns come BEFORE "sand" to catch manufactured-sand-in-mortar
#   - "waterproofing" comes BEFORE "paint" (primer coat overlap)
#   - "formwork" added as its own category
#   - "flooring" added as its own category
# ─────────────────────────────────────────────────────────────────────────────

# Patterns for SECTION HEADERS (zero-qty parent rows)
# These set the current material context for child rows below them
SECTION_HEADER_PATTERNS = [
    # (material_name, [regex patterns on lowercased description])
    ("concrete",    [r'reinforced.{0,15}cement.{0,15}concrete',
                     r'plain.{0,15}cement.{0,15}concrete',
                     r'\brcc\b', r'\bpcc\b', r'ready.?mix.{0,10}concrete', r'\brmc\b',
                     r'cement concrete works']),
    ("steel",       [r'steel reinforcement works', r'tmt bar.*supply', r'supply.*tmt']),
    ("formwork",    [r'aluminium system formwork', r'system formwork', r'mivan',
                     r'centering.*shuttering', r'shuttering.*centering',
                     r'formwork works']),
    ("waterproofing",[r'waterproof(?:ing)? works', r'waterproof(?:ing)? treatment',
                      r'waterproof(?:ing)? to.*roof', r'waterproof(?:ing)? for.*basement']),
    ("flooring",    [r'flooring works', r'floor.*finish', r'tile.*flooring',
                     r'granite.*flooring', r'marble.*flooring', r'kota.*stone',
                     r'vdf.*flooring', r'ips.*flooring']),
    ("paint",       [r'painting works', r'painting and finishing', r'exterior.*paint',
                     r'internal.*paint']),
    ("plaster",     [r'plaster(?:ing)? works', r'internal.*plaster', r'external.*plaster']),
    ("masonry",     [r'masonry works', r'brick.*work', r'block.*masonry']),
    ("aluminium",   [r'aluminium.*works', r'curtain wall', r'facade.*works']),
    ("excavation",  [r'earth works', r'excavation works']),
    ("anti_termite",[r'anti.?termite', r'termite treatment']),
]

# Patterns for INDIVIDUAL ROWS (any qty, matching the description itself)
# Tried only when section-header context is insufficient or row has full description
ROW_PATTERNS = [
    # concrete — explicit descriptions
    ("concrete",    [r'providing and laying.*concrete', r'reinforced cement concrete',
                     r'plain cement concrete', r'\bplain\b.*\bcement\b.*\bconcrete\b',
                     r'lean concrete', r'plum concrete',
                     r'\brcc\b', r'\bpcc\b', r'\br\.c\.c\b', r'p\.c\.c\.',
                     r'concrete of grade', r'grade.*concrete', r'in.?situ concrete',
                     r'readymix concrete', r'ready mix concrete', r'ready.?mixed concrete',
                     r'\brmc\b', r'screed concrete', r'batch mixed.*concrete',
                     r'providing.*concrete.*grade', r'grade.*m\d{2}.*concrete']),

    # pipe — MUST come before steel to catch conduit correctly
    ("pipe",        [r'upvc.*pipe', r'cpvc.*pipe', r'pvc.*pipe',
                     r'swr.*pipe', r'gi.*pipe', r'hdpe.*pipe',
                     r'conduit.*pipe', r'frls.*conduit', r'heavy duty.*conduit',
                     r'drainage.*pipe', r'supply.*pipe.*dia',
                     r'puddle.*flange', r'\bflange\b.*pipe',
                     r'rcc.*np.*class.*pipe', r'subsoil.*drain.*pipe']),

    # steel — rebar/structural only (after pipe so conduit is caught first)
    ("steel",       [r'steel reinforcement', r'tmt.*bar', r'tor.*steel',
                     r'\brebars?\b', r'high yield.*bar', r'hysd.*bar', r'hsd.*bar',
                     r'mild steel.*bar', r'\bms\b.*\bbar\b',
                     r'welded.*mesh', r'wire.*mesh', r'structural steel',
                     r'\bms\b.*\bangle\b', r'\bms\b.*\bchannel\b',
                     r'binding wire', r'bar bending',
                     r'reinforcement.*fe\s?\d{3}', r'fe\s?500.*ton', r'fe\s?415.*ton']),

    # tile/granite flooring — MUST come before sand
    # 'manufactured sand' appears in mortar bed specs; the item is actually flooring
    ("tile",        [r'ceramic tile', r'vitrified tile', r'mosaic tile',
                     r'granite.*floor', r'marble.*floor', r'kota stone',
                     r'tile.*flooring', r'floor.*tile', r'dado.*tile',
                     r'skirting.*tile', r'natural stone.*floor',
                     r'clay tile', r'lift car.*floor',
                     r'providing.*laying.*tile', r'providing.*laying.*granite',
                     r'providing.*laying.*marble']),

    # sand — raw material supply only (after tile)
    ("sand",        [r'supply.*fine aggregate', r'supply.*river sand',
                     r'supply.*m.?sand', r'supply.*manufactured sand',
                     r'supply.*robo sand',
                     r'fine aggregate.*supply', r'sand.*supply']),

    # waterproofing — before paint (primer coat overlap)
    ("waterproofing",[r'waterproof(?:ing)?', r'integral waterproof',
                      r'torch applied', r'crystalline waterproof',
                      r'cementitious waterproof', r'\bdpc\b',
                      r'dampproof', r'damp proof', r'pu.*membrane',
                      r'polyurethane.*membrane', r'bituminous.*membrane']),

    # paint — after waterproofing
    ("paint",       [r'\bpainting\b', r'emulsion paint', r'enamel paint',
                     r'\bwhitewash\b', r'\bdistemper\b', r'texture.*coat',
                     r'weather shield', r'acrylic.*paint',
                     r'exterior paint', r'interior paint', r'wall paint',
                     r'obd\b', r'plastic emulsion']),

    # formwork — shuttering/centering items
    ("formwork",    [r'aluminium system formwork', r'aluminium.*shuttering',
                     r'mivan.*shuttering', r'centering.*formwork',
                     r'formwork.*centering', r'extra.*staging',
                     r'double.*height.*staging']),

    # flooring — composite floor finishes (not raw tile)
    ("flooring",    [r'vacuum.*dewat.*floor', r'vdf.*floor', r'ips.*floor',
                     r'power.*float.*floor', r'floor hardener',
                     r'granolithic.*floor', r'providing.*laying.*flooring']),

    # remaining categories
    ("brick",       [r'brick work', r'brick masonry', r'fly ash brick',
                     r'clay brick', r'aac block', r'hollow block',
                     r'block masonry', r'\bbrick\b', r'burnt clay',
                     r'wire cut brick']),
    ("timber",      [r'\btimber\b', r'\bplywood\b', r'door frame', r'window frame',
                     r'wooden.*door', r'teak wood', r'hardwood']),
    ("glass",       [r'toughened glass', r'tempered glass', r'float glass',
                     r'laminated glass', r'double glazed', r'\bglazing\b',
                     r'glass.*panel', r'glass.*partition', r'lacquered glass']),
    ("aggregate",   [r'coarse aggregate', r'crushed stone',
                     r'20mm.*metal', r'40mm.*metal', r'stone chips', r'\bjelly\b']),
    ("plaster",     [r'cement plaster', r'gypsum plaster', r'sand face plaster',
                     r'\bplastering\b', r'neeru finish', r'skim coat', r'rendering']),
    ("mortar",      [r'cement mortar', r'pointing mortar', r'bedding mortar']),
    ("excavation",  [r'excavation in', r'excavation for', r'earth work.*excavat',
                     r'backfill(?:ing)?', r'filling.*foundation',
                     r'carting.*surplus earth', r'cinder.*fill']),
    ("anti_termite",[r'anti.?termite', r'termite treatment',
                     r'pre.?construction.*termite']),
    ("insulation",  [r'thermal insulation', r'rock wool', r'mineral wool',
                     r'\bxps\b', r'\beps\b.*board', r'thermocoal.*insul']),
    ("aluminium",   [r'\baluminium\b', r'\baluminum\b', r'alum.*section',
                     r'aluminium composite', r'\bacp\b.*panel', r'curtain wall',
                     r'hpl.*sheet', r'aluminium.*fin', r'aluminium.*louver',
                     r'aluminium.*profile', r'aluminium.*cover']),
    ("sealant",     [r'polysulphide sealant', r'pu.*sealant', r'joint.*sealant',
                     r'expansion joint.*seal', r'sealant.*joint',
                     r'kantaflex', r'masterflex', r'colpor']),
    ("grouting",    [r'non.?shrink grout', r'cebex.*grout', r'epoxy.*grout',
                     r'injection.*grout', r'pu.*grout']),
]

# Section breaks: keywords that signal a COMPLETELY different trade section
# When hit, both current_material and current_grade are reset
# NOTE: "formwork" and "shuttering" deliberately excluded —
#       they appear mid-section within concrete work packages
SECTION_BREAK_KEYWORDS = [
    "painting works", "painting and finishing",
    "steel reinforcement works",
    "waterproofing works",
    "tile flooring works", "flooring works",
    "plumbing works", "sanitary works",
    "electrical works", "hvac works",
    "landscaping works", "external development",
    "false ceiling works", "interior works",
]

# Header rows that signal we are inside a concrete context
CONCRETE_SECTION_PAT = re.compile(
    r'(reinforced.{0,15}cement.{0,15}concrete'
    r'|plain.{0,15}cement.{0,15}concrete'
    r'|\brcc\b|\bpcc\b'
    r'|ready.?mix.{0,10}concrete|\brmc\b)',
    re.IGNORECASE
)


def detect_material_from_row(desc_clean: str):
    """Try to detect material from a row's own description text."""
    for material, patterns in ROW_PATTERNS:
        for pat in patterns:
            if re.search(pat, desc_clean):
                return material
    return None


def detect_material_from_section(desc_clean: str):
    """Try to detect material from a section header description."""
    for material, patterns in SECTION_HEADER_PATTERNS:
        for pat in patterns:
            if re.search(pat, desc_clean):
                return material
    return None


def is_section_break(desc_clean: str) -> bool:
    for kw in SECTION_BREAK_KEYWORDS:
        if kw in desc_clean:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — FILE LOADING
# Each file has different column layout — configured explicitly
# ─────────────────────────────────────────────────────────────────────────────

FILE_CONFIGS = [
    # (filename,  sheet_name,                      skip, desc_col, unit_col, qty_col, rate_col, source)
    ("Bot.xlsx",  "Part-A & B_BOQ",                1,    1,        2,        3,        None,    "Bot"),
    ("Eco.xlsx",  "Part-A_Structure+Archi BOQ",    0,    1,        2,        6,        None,    "Eco"),
    ("Mall.xlsx", "BOQ",                           1,    1,        2,        3,        None,    "Mall"),
    ("Zen.xlsx",  "Abstract-Str& Fin-PH02",        0,    1,        2,        3,        None,    "Zen"),
    ("PA.xlsx",   "Abstract",                      0,    2,        3,        5,        4,       "PA"),
]

JUNK_PAT = re.compile(
    r'^(total|sub.?total|grand total|subtotal|carry forward|brought forward'
    r'|sub total|page total|amount)',
    re.IGNORECASE
)


def load_raw_file(cfg) -> pd.DataFrame:
    fname, sheet, skip, desc_col, unit_col, qty_col, rate_col, source = cfg
    path = RAW / fname
    df = pd.read_excel(path, sheet_name=sheet, header=None, skiprows=skip)
    rows = []
    for _, row in df.iterrows():
        desc = row.iloc[desc_col] if desc_col < len(row) else None
        unit = row.iloc[unit_col] if unit_col < len(row) else None
        qty  = row.iloc[qty_col]  if qty_col  < len(row) else None
        rate = row.iloc[rate_col] if (rate_col is not None and rate_col < len(row)) else None
        rows.append({
            "source":      source,
            "description": desc,
            "unit_raw":    unit,
            "quantity_raw": qty,
            "rate_raw":    rate,
        })
    df_out = pd.DataFrame(rows)
    print(f"  Loaded {fname}: {len(df_out):,} rows")
    return df_out


def load_all_files() -> pd.DataFrame:
    print("\n[1] Loading raw files...")
    frames = [load_raw_file(cfg) for cfg in FILE_CONFIGS]
    all_raw = pd.concat(frames, ignore_index=True)
    print(f"  Total raw rows: {len(all_raw):,}")
    return all_raw


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — PROPAGATION ENGINE
# Must run on the raw (unfiltered) DataFrame so zero-qty header rows
# are present to provide context for child rows.
# ─────────────────────────────────────────────────────────────────────────────

def propagate_context(all_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Walk every row in file order, per source.
    Maintain:
      current_grade    — concrete grade (M5–M50), reset at section breaks
      current_material — material category from section header
      in_concrete_sec  — True when inside an RCC/PCC section

    Store results as:
      prop_grade    — grade propagated to this row from a parent header
      prop_material — material propagated from a section header
    """
    print("\n[2] Propagating grade and material context...")
    all_raw = all_raw.copy()
    all_raw["desc_clean_raw"] = all_raw["description"].apply(clean_text)

    grade_map    = {}
    material_map = {}
    in_conc_map  = {}

    for src in ["Bot", "Eco", "Mall", "Zen", "PA"]:
        sub = all_raw[all_raw["source"] == src]
        current_grade    = None
        current_material = None
        in_concrete_sec  = False

        for idx in sub.index:
            desc = sub.at[idx, "desc_clean_raw"]

            # ── Section break resets everything ──────────────────────────────
            if is_section_break(desc):
                current_grade    = None
                current_material = None
                in_concrete_sec  = False

            # ── Detect concrete section header ────────────────────────────────
            raw_desc = str(sub.at[idx, "description"]) if pd.notna(sub.at[idx, "description"]) else ""
            if CONCRETE_SECTION_PAT.search(raw_desc):
                in_concrete_sec  = True
                current_material = "concrete"

            # ── Detect material from section header ───────────────────────────
            mat_from_header = detect_material_from_section(desc)
            if mat_from_header:
                current_material = mat_from_header
                if mat_from_header == "concrete":
                    in_concrete_sec = True
                elif mat_from_header != "concrete":
                    # Only reset concrete context if entering a clearly different section
                    if mat_from_header not in ("formwork", "waterproofing"):
                        in_concrete_sec = False
                        current_grade   = None

            # ── Detect concrete grade anywhere in the row ─────────────────────
            g = extract_concrete_grade(desc)
            if g:
                current_grade   = g
                in_concrete_sec = True
                current_material = "concrete"

            grade_map[idx]    = current_grade
            material_map[idx] = current_material
            in_conc_map[idx]  = in_concrete_sec

    all_raw["prop_grade"]    = pd.Series(grade_map,    dtype=object)
    all_raw["prop_material"] = pd.Series(material_map, dtype=object)
    all_raw["in_conc_sec"]   = pd.Series(in_conc_map,  dtype=bool)

    grade_rows = all_raw["prop_grade"].notna().sum()
    mat_rows   = all_raw["prop_material"].notna().sum()
    print(f"  Rows with propagated grade:    {grade_rows:,}")
    print(f"  Rows with propagated material: {mat_rows:,}")
    return all_raw


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — ROW CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_rows(all_raw: pd.DataFrame) -> pd.DataFrame:
    """
    1. Drop rows with no description
    2. Convert quantity to numeric; drop NaN and non-positive
       (handles 'RO' = Rate Only in PA/Zen)
    3. Drop total/subtotal/junk header rows
    4. Deduplicate within each source
    5. Standardise units
    6. Convert rate to numeric (PA only)
    7. Apply NLP text cleaning
    """
    print("\n[3] Cleaning rows...")
    raw_n = len(all_raw)

    df = all_raw[all_raw["description"].notna()].copy()
    df["quantity"] = pd.to_numeric(df["quantity_raw"], errors="coerce")
    df = df[df["quantity"].notna() & (df["quantity"] > 0)]

    desc_tmp  = df["description"].apply(clean_text)
    junk_mask = desc_tmp.str.contains(JUNK_PAT, regex=True, na=False)
    df = df[~junk_mask]

    df.drop_duplicates(subset=["description", "source"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["unit_clean"]        = df["unit_raw"].apply(standardise_unit)
    df["rate"]              = pd.to_numeric(df["rate_raw"], errors="coerce")
    df["description_clean"] = df["description"].apply(clean_text_nlp)

    print(f"  {raw_n:,} raw → {len(df):,} clean rows  ({raw_n - len(df):,} removed)")
    for src in ["Bot", "Eco", "Mall", "Zen", "PA"]:
        n = (df["source"] == src).sum()
        print(f"    {src}: {n:,} rows")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — MATERIAL ASSIGNMENT
# Layer 1: keyword match on the row's own description
# Layer 2: inherit from propagated section context
# Layer 3: for concrete — unit=m3 + grade context = concrete
# ─────────────────────────────────────────────────────────────────────────────

def assign_materials(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns material, grade, and a thickness_mm column to every row.
    """
    print("\n[4] Assigning materials and grades...")

    materials  = []
    grades     = []
    thicknesses = []
    extr_methods = []

    for _, row in df.iterrows():
        desc      = str(row["description_clean"])
        desc_raw  = str(row["description"]).lower()
        unit      = str(row["unit_clean"])
        pg        = row["prop_grade"]
        pm        = row["prop_material"]
        in_conc   = bool(row["in_conc_sec"])

        # ── Layer 1: keyword on this row's own description ────────────────────
        mat = detect_material_from_row(desc)
        method = "keyword" if mat else None

        # ── Layer 2: concrete reclassification via grade propagation ──────────
        # Any m3 row with a propagated concrete grade = concrete
        if mat is None or mat == "concrete":
            if pd.notna(pg) and unit == "m3":
                mat    = "concrete"
                method = "grade_propagation"

        # ── Layer 3: in concrete section + m3 = concrete ─────────────────────
        if mat is None and in_conc and unit == "m3":
            mat    = "concrete"
            method = "section_context"

        # ── Layer 4: m2 screed/slab rows under a concrete grade context ───────
        if mat is None and pd.notna(pg) and unit == "m2":
            slab_kws = ["slab", "screed", "raft", "grade slab", "floor slab",
                        "protective layer", "protective screed"]
            if any(kw in desc for kw in slab_kws):
                mat    = "concrete"
                method = "grade_propagation_m2"

        # ── Layer 5: inherit from propagated material (section header) ─────────
        if mat is None and pd.notna(pm):
            mat    = pm
            method = "section_context"

        # ── Grade assignment ───────────────────────────────────────────────────
        grade = None
        if mat == "concrete":
            g_explicit = extract_concrete_grade(desc)
            if g_explicit:
                grade = g_explicit
            elif pd.notna(pg):
                grade = pg
            else:
                grade = "M20"   # default — most common Indian structural grade
                method = method + "_defaultgrade" if method else "default_grade"

        elif mat == "steel":
            g = extract_steel_grade(desc)
            if g:
                grade = g
            elif any(kw in desc for kw in ["tmt", "fe500", "500d"]):
                grade = "Fe500"
            elif any(kw in desc for kw in ["fe415", "hsd", "hysd"]):
                grade = "Fe415"
            elif any(kw in desc for kw in ["mild steel", "fe250", "ms bar"]):
                grade = "Fe250"
            else:
                grade = "Fe500"  # default TMT

        # ── Thickness extraction for m2 concrete rows ─────────────────────────
        thk = None
        if mat == "concrete" and unit == "m2":
            thk = extract_thickness_mm(desc_raw)
            if thk is None:
                thk = extract_thickness_mm(desc)

        materials.append(mat if mat else "UNKNOWN")
        grades.append(grade)
        thicknesses.append(thk)
        extr_methods.append(method if method else "unmatched")

    df["material"]     = materials
    df["grade"]        = grades
    df["thickness_mm"] = thicknesses
    df["extr_method"]  = extr_methods

    known_mat = (df["material"] != "UNKNOWN").sum()
    concrete  = (df["material"] == "concrete").sum()
    graded    = df[df["material"] == "concrete"]["grade"].notna().sum()
    m2_conc   = ((df["material"] == "concrete") & (df["unit_clean"] == "m2")).sum()
    m2_thk    = ((df["material"] == "concrete") & (df["unit_clean"] == "m2") & df["thickness_mm"].notna()).sum()

    print(f"  Material coverage : {known_mat}/{len(df)} ({known_mat/len(df)*100:.1f}%)")
    print(f"  Concrete rows     : {concrete}")
    print(f"  Graded concrete   : {graded} ({graded/concrete*100:.1f}%)" if concrete else "")
    print(f"  m2 concrete rows  : {m2_conc} | thickness extracted: {m2_thk}")

    print(f"\n  Material distribution:")
    for mat, cnt in df["material"].value_counts().items():
        print(f"    {mat:20s}: {cnt:4d}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — SAVE OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

FINAL_COLS = [
    "source",
    "description",
    "description_clean",
    "unit_clean",
    "quantity",
    "rate",           # populated for PA rows only; NaN for others
    "material",
    "grade",
    "thickness_mm",   # populated for m2 concrete rows only
    "extr_method",
    "prop_grade",
    "prop_material",
    "in_conc_sec",
]


def save_output(df: pd.DataFrame):
    out_path = PROC / "master_cleaned.csv"
    df[FINAL_COLS].to_csv(out_path, index=False)
    print(f"\n[5] Saved → {out_path}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {FINAL_COLS}")

    # Quick sanity check
    pa_rows = df[df["source"] == "PA"]
    pa_rated = pa_rows["rate"].notna().sum()
    print(f"\n  PA rows with rate: {pa_rated} / {len(pa_rows)}")
    print(f"  Rate range (PA)  : INR {pa_rows['rate'].min():.0f} – {pa_rows['rate'].max():.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — QUALITY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def quality_report(df: pd.DataFrame, raw_n: int):
    print("\n" + "=" * 60)
    print("PART 1 QUALITY REPORT")
    print("=" * 60)

    total    = len(df)
    conc     = df[df["material"] == "concrete"]
    steel    = df[df["material"] == "steel"]
    coverage = (df["material"] != "UNKNOWN").mean()

    print(f"  Raw rows loaded       : {raw_n:,}")
    print(f"  Clean rows            : {total:,}  ({raw_n - total:,} removed)")
    print(f"  Material coverage     : {coverage*100:.1f}%")
    print(f"\n  Concrete rows         : {len(conc)}")
    if len(conc):
        print(f"    Grade fill          : {conc['grade'].notna().mean()*100:.1f}%")
        print(f"    Grade distribution  :")
        for g, c in conc["grade"].value_counts().items():
            print(f"      {g:5s}: {c:4d} rows")
        m2 = conc[conc["unit_clean"] == "m2"]
        print(f"    m2 concrete rows    : {len(m2)}")
        print(f"    thickness extracted : {m2['thickness_mm'].notna().sum()}")

    print(f"\n  Steel rows            : {len(steel)}")
    if len(steel):
        for g, c in steel["grade"].value_counts().items():
            print(f"    {g}: {c} rows")

    print(f"\n  Per-source summary:")
    for src in ["Bot", "Eco", "Mall", "Zen", "PA"]:
        s = df[df["source"] == src]
        c = s[s["material"] == "concrete"]
        r = s["rate"].notna().sum()
        print(f"    {src:4s}: {len(s):3d} rows | concrete: {len(c):3d} | "
              f"graded: {c['grade'].notna().sum()}/{len(c)} | "
              f"rated: {r}")

    print(f"\n  Extraction method breakdown:")
    for m, cnt in df["extr_method"].value_counts().items():
        print(f"    {m:35s}: {cnt}")

    # Assertions
    print(f"\n  Assertions:")
    assert df["description"].isna().sum() == 0, "Empty descriptions found"
    print("  OK  No empty descriptions")
    assert (df["quantity"] > 0).all(), "Non-positive quantities found"
    print("  OK  All quantities > 0")
    assert len(conc) > 100, f"Too few concrete rows: {len(conc)}"
    print(f"  OK  Concrete rows > 100 ({len(conc)})")
    assert conc["grade"].notna().mean() > 0.90, \
        f"Grade fill rate too low: {conc['grade'].notna().mean()*100:.1f}%"
    print(f"  OK  Concrete grade fill > 90% ({conc['grade'].notna().mean()*100:.1f}%)")

    # No electrical conduit in steel
    steel_conduit = steel[steel["description"].str.contains(
        r'conduit|frls', case=False, na=False)]
    assert len(steel_conduit) == 0, \
        f"Conduit rows still in steel: {len(steel_conduit)}"
    print("  OK  No electrical conduit misclassified as steel")

    # No tile/granite rows in sand
    sand_rows = df[df["material"] == "sand"]
    sand_tile = sand_rows[sand_rows["description"].str.contains(
        r'granite|tile|marble', case=False, na=False)]
    assert len(sand_tile) == 0, \
        f"Tile rows still in sand: {len(sand_tile)}"
    print("  OK  No tile/granite rows misclassified as sand")

    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("PART 1 — DATA INGESTION AND CLEANING")
    print("=" * 60)

    all_raw = load_all_files()
    raw_n   = len(all_raw)

    all_raw = propagate_context(all_raw)   # must run before cleaning
    df      = clean_rows(all_raw)
    df      = assign_materials(df)

    quality_report(df, raw_n)
    save_output(df)

    print("\nPart 1 complete.")
    print("Next: run src/part2_emission_factors.py")
    return df


if __name__ == "__main__":
    main()