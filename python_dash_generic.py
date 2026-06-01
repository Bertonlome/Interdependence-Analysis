"""
Generic Interdependence Analysis Dashboard
© Benjamin R. Berton 2025 Polytechnique Montreal

Parametric for any Human-Autonomy Team configuration.
Auto-detects team structure from CSV or allows manual definition.
"""
import dash
from dash import html, dcc, dash_table, Input, Output, State, callback_context
import plotly.graph_objects as go
import pandas as pd
import os
import base64
import io
import textwrap
import time
import pathlib

# Path to the bundled example CSV (works locally and in deployment)
_HERE = pathlib.Path(__file__).parent
_EXAMPLE_CANDIDATES = [_HERE / "V8" / "IA_V8.csv", _HERE / "IA_V8.csv"]
EXAMPLE_CSV = next((p for p in _EXAMPLE_CANDIDATES if p.exists()), None)

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=["assets/styles.css"])
server = app.server

# ─── Design Palette ────────────────────────────────────────────────────────────
# Edit these to re-skin all graphs / inline styles in one place.
BG           = "#ffffff"
SURFACE      = "#f7f7f5"
SURFACE      = "#f7f7f5"
INK          = "#1a1a1a"
INK_MUTED    = "#313131"
BORDER       = "#b0ada6"
ACCENT       = "#d60b0b"      # also used for highlights
DARK_GREY = "#1F1F1F"

# Semantic colours (mapped from the IA red/yellow/green/orange scheme)
PAL_RED      = "#d60b0b"
PAL_ORANGE   = "#d9510c"
PAL_YELLOW   = "#fae608"
PAL_GREEN    = "#029c3d"
# Map used by style_table() and dot colours
COLOR_MAP = {
    "red":    PAL_RED,
    "orange": PAL_ORANGE,
    "yellow": PAL_YELLOW,
    "green":  PAL_GREEN,
}
# Lighter variants for pie charts / secondary usage
COLOR_MAP_LIGHT = {
    "red":    "#d44040",
    "orange": "#de7642",
    "yellow": "#f5e74e",
    "green":  "#1b7f41",
}
# Shade families for grouped bar charts (capacity charts)
COLOR_SHADES = {
    "green":  [PAL_GREEN, "#1b7f41", "#2a8f52", "#3a9f63", "#4abf74", "#4f8a5c"],
    "yellow": [PAL_YELLOW, "#f5e74e", "#f7f9a8", "#f9fbc4", "#fbe98a", "#c9b43e"],
    "orange": [PAL_ORANGE, "#d88a5c", "#e09c73", "#b45a2e", "#c47648", "#a04e22"],
}

# ─── Constants ─────────────────────────────────────────────────────────────────
COLOR_OPTIONS = ["red", "yellow", "green", "orange"]
VALID_COLORS = {"red", "yellow", "green", "orange"}


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def wrap_text(text, max_width=30):
    if not isinstance(text, str):
        return ""
    return "<br>".join(textwrap.wrap(text, width=max_width))


# ═══════════════════════════════════════════════════════════════════════════════
# TEAM CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_team_config(df):
    """
    Auto-detect team configuration from a DataFrame.

    Identifies color-valued columns as agent role columns, then groups them
    into team alternatives using a state machine that detects transitions
    between performer (*) and supporter (no *) column groups.

    Returns a config dict or None if detection fails.
    """
    # 1. Find columns whose values are mostly valid colors
    color_columns = []
    for col in df.columns:
        vals = df[col].dropna().astype(str).str.strip().str.lower()
        if len(vals) > 0 and vals.isin(VALID_COLORS).sum() / len(vals) > 0.3:
            color_columns.append(col)

    if not color_columns:
        return None

    # 2. Parse team alternatives with a state machine
    #    Rule: consecutive performer columns (*) form one group, then consecutive
    #    supporter columns form another. When we see a performer after supporters,
    #    a new alternative begins.
    alternatives = []
    current_alt = {"performers": [], "supporters": []}
    state = "start"

    for col in color_columns:
        is_performer = col.endswith("*")
        if is_performer:
            if state == "in_supporters":
                # Transition supporter→performer = new alternative
                alternatives.append(current_alt)
                current_alt = {"performers": [], "supporters": []}
            current_alt["performers"].append(col)
            state = "in_performers"
        else:
            current_alt["supporters"].append(col)
            state = "in_supporters"

    if current_alt["performers"] or current_alt["supporters"]:
        alternatives.append(current_alt)

    # 3. Extract unique agent base names (preserving first-seen order)
    agent_names = []
    seen = set()
    for alt in alternatives:
        for p in alt["performers"]:
            base = p.rstrip("*")
            if base not in seen:
                agent_names.append(base)
                seen.add(base)
        for s in alt["supporters"]:
            if s not in seen:
                agent_names.append(s)
                seen.add(s)

    agents = []
    for name in agent_names:
        atype = "human" if name.lower() in ["human", "pilot", "operator", "crew"] else "autonomous"
        agents.append({"name": name, "type": atype})

    # 4. Detect hierarchy columns: structured levels before first color column.
    #    Strategy:
    #    (a) Find all structural candidates (non-color, non-Row before first color col)
    #    (b) Mark "category-like" columns: keyword match + few unique values
    #    (c) Detect procedure_column (keyword) and task_column (keyword or cardinality)
    #    (d) hierarchy = structural columns from procedure to task, excluding category-like
    color_set = set(color_columns)
    col_list = list(df.columns)
    first_color_idx = min((col_list.index(c) for c in color_columns), default=len(col_list))
    structural_candidates = [
        col for i, col in enumerate(col_list)
        if col != "Row" and col not in color_set and i < first_color_idx
    ]
    structural_set_all = set(structural_candidates)

    # (b) Category-like: column name matches a classification keyword AND has few unique values
    CATEGORY_KEYWORDS = {"Category", "Type", "Classification", "Class", "Stage"}
    category_like = {
        c for c in structural_candidates
        if c in CATEGORY_KEYWORDS and df[c].dropna().nunique() <= max(10, len(df) * 0.15)
    }

    # (c) Procedure column: keyword-based
    procedure_column = None
    for candidate in ["Procedure", "procedure", "Phase", "Group", "Section"]:
        if candidate in structural_set_all:
            procedure_column = candidate
            break
    if procedure_column is None and structural_candidates:
        procedure_column = structural_candidates[0]

    # (c) Task column: keyword-based, then fallback to last structural candidate
    #     (In a hierarchical CSV, the most specific level is defined last,
    #      closest to the color/assessment columns.)
    task_column = None
    for candidate in ["Task Object", "Task"]:
        if candidate in structural_set_all:
            task_column = candidate
            break
    if task_column is None and structural_candidates:
        task_column = structural_candidates[-1]

    # (d) Build hierarchy slice: from procedure to task (inclusive), excluding category-like
    try:
        proc_idx = col_list.index(procedure_column) if procedure_column else 0
        task_idx = col_list.index(task_column) if task_column else len(col_list) - 1
        if proc_idx <= task_idx:
            hier_slice = [
                c for c in col_list[proc_idx:task_idx + 1]
                if c != "Row" and c not in color_set
            ]
        else:
            hier_slice = [c for c in [procedure_column, task_column] if c]
    except (ValueError, TypeError):
        hier_slice = [c for c in structural_candidates if c not in category_like]
    hierarchy_columns = [c for c in hier_slice if c not in category_like][:4]

    # Ensure procedure and task are always represented in the hierarchy
    if procedure_column and procedure_column not in hierarchy_columns and procedure_column in structural_set_all:
        hierarchy_columns.insert(0, procedure_column)
    if task_column and task_column not in hierarchy_columns and task_column in structural_set_all:
        hierarchy_columns.append(task_column)
    hierarchy_columns = hierarchy_columns[:4]

    # Update back-compat aliases to reflect final hierarchy
    procedure_column = hierarchy_columns[0] if hierarchy_columns else None
    task_column = hierarchy_columns[-1] if hierarchy_columns else None

    # 4b. Detect the category column (from category-like columns, for Automation Proportion)
    hier_set = set(hierarchy_columns)
    category_column = None
    for candidate in ["Category", "Type", "Classification", "Class", "Stage"]:
        if candidate in df.columns and candidate in category_like:
            category_column = candidate
            break
    if category_column is None:
        # Fallback: first non-hierarchy column with few unique values
        hier_set = set(hierarchy_columns)
        for col in df.columns:
            if col in hier_set or col in color_set or col == "Row":
                continue
            vals = df[col].dropna().astype(str)
            if 1 < vals.nunique() <= max(10, len(df) * 0.2):
                category_column = col
                break

    # 5. Identify metadata columns (everything not structural/agent/category)
    structural = {"Row"} | hier_set
    if category_column:
        structural.add(category_column)
    metadata = [c for c in df.columns if c not in structural and c not in color_set]

    config = {
        "agents": agents,
        "alternatives": [
            {"name": f"Team Alternative {i+1}", **alt}
            for i, alt in enumerate(alternatives)
        ],
        "task_column": task_column or "Task",
        "procedure_column": procedure_column or "Procedure",
        "hierarchy_columns": hierarchy_columns,
        "category_column": category_column,
        "color_columns": color_columns,
        "metadata_columns": metadata,
        "all_columns": list(df.columns),
    }
    return config


def build_config_from_manual(column_str, task_col="Task", procedure_col="Procedure", category_col=None):
    """
    Build team config from a manual column specification string.

    Example input: "Human*, TARS, TARS*, Human"
    This will be parsed with the same state machine as CSV detection.
    """
    cols = [c.strip() for c in column_str.split(",") if c.strip()]
    if not cols:
        return None

    # Parse alternatives
    alternatives = []
    current_alt = {"performers": [], "supporters": []}
    state = "start"

    for col in cols:
        is_perf = col.endswith("*")
        if is_perf:
            if state == "in_supporters":
                alternatives.append(current_alt)
                current_alt = {"performers": [], "supporters": []}
            current_alt["performers"].append(col)
            state = "in_performers"
        else:
            current_alt["supporters"].append(col)
            state = "in_supporters"

    if current_alt["performers"] or current_alt["supporters"]:
        alternatives.append(current_alt)

    # Extract agents
    agent_names = []
    seen = set()
    for alt in alternatives:
        for p in alt["performers"]:
            base = p.rstrip("*")
            if base not in seen:
                agent_names.append(base)
                seen.add(base)
        for s in alt["supporters"]:
            if s not in seen:
                agent_names.append(s)
                seen.add(s)

    agents = []
    for name in agent_names:
        atype = "human" if name.lower() in ["human", "pilot", "operator", "crew"] else "autonomous"
        agents.append({"name": name, "type": atype})

    extra_cols = ["Observability", "Predictability", "Directability"]
    struct_cols = ["Row", procedure_col, task_col]
    if category_col and category_col not in struct_cols:
        struct_cols.insert(3, category_col)  # insert after task_col
    all_columns = struct_cols + cols + extra_cols
    # For manual setup, hierarchy = [procedure, task] (2-level)
    hier_cols = [c for c in [procedure_col, task_col] if c]

    config = {
        "agents": agents,
        "alternatives": [
            {"name": f"Team Alternative {i+1}", **alt}
            for i, alt in enumerate(alternatives)
        ],
        "task_column": task_col,
        "procedure_column": procedure_col,
        "hierarchy_columns": hier_cols,
        "category_column": category_col,
        "color_columns": cols,
        "metadata_columns": extra_cols,
        "all_columns": all_columns,
    }
    return config


# ─── Config helper accessors ──────────────────────────────────────────────────

def get_agent_columns(config):
    """Ordered list of all agent (color) columns."""
    return config.get("color_columns", [])


def get_performer_columns(config):
    """All performer columns (ending with *) across all alternatives."""
    out = []
    for alt in config.get("alternatives", []):
        out.extend(alt["performers"])
    return out


def get_supporter_columns(config):
    """All supporter columns (no *) across all alternatives."""
    out = []
    for alt in config.get("alternatives", []):
        out.extend(alt["supporters"])
    return out


def get_chosen_performer(row, config, strategy, category_overrides=None):
    """
    Determine the chosen performer column for a task row based on the strategy.

    Strategies:
    - human_baseline / human_full_support: prefer human-type performers
    - agent_whenever_possible / agent_whenever_possible_full_support: prefer autonomous performers
    - most_reliable: best color (green > yellow), human preferred in ties

    Returns column name (e.g. "Human*") or None.
    """
    performer_cols = get_performer_columns(config)
    agent_types = {a["name"]: a["type"] for a in config["agents"]}
    COLOR_PRIORITY = {"green": 1, "yellow": 2, "orange": 3}

    # Independent strategies cannot use orange performers (orange = forced interdependence)
    independent_strategy = strategy in ("human_baseline", "agent_whenever_possible")

    # Gather available performers (non-red; orange excluded on independent paths)
    available = {}
    for pc in performer_cols:
        val = str(row.get(pc, "") or "").strip().lower()
        if val in VALID_COLORS and val != "red":
            if independent_strategy and val == "orange":
                continue
            available[pc] = val

    if not available:
        return None

    # Category override check
    if category_overrides:
        cat_col = config.get("category_column") or "Category"
        cat = str(row.get(cat_col, "") or "").strip()
        if cat in category_overrides:
            override_type = category_overrides[cat].lower()  # "human" or "autonomous"
            preferred = {
                pc: c for pc, c in available.items()
                if agent_types.get(pc.rstrip("*"), "").lower() == override_type
            }
            if preferred:
                return min(preferred, key=lambda pc: COLOR_PRIORITY.get(preferred[pc], 999))
            return min(available, key=lambda pc: COLOR_PRIORITY.get(available[pc], 999))

    if strategy in ("human_baseline", "human_full_support"):
        human_perfs = {
            pc: c for pc, c in available.items()
            if agent_types.get(pc.rstrip("*"), "").lower() == "human"
        }
        if human_perfs:
            return min(human_perfs, key=lambda pc: COLOR_PRIORITY.get(human_perfs[pc], 999))
        return None

    elif strategy in ("agent_whenever_possible", "agent_whenever_possible_full_support"):
        auto_perfs = {
            pc: c for pc, c in available.items()
            if agent_types.get(pc.rstrip("*"), "").lower() == "autonomous"
        }
        if auto_perfs:
            return min(auto_perfs, key=lambda pc: COLOR_PRIORITY.get(auto_perfs[pc], 999))
        human_perfs = {
            pc: c for pc, c in available.items()
            if agent_types.get(pc.rstrip("*"), "").lower() == "human"
        }
        if human_perfs:
            return min(human_perfs, key=lambda pc: COLOR_PRIORITY.get(human_perfs[pc], 999))
        return None

    elif strategy == "most_reliable":
        def sort_key(pc):
            cprio = COLOR_PRIORITY.get(available[pc], 999)
            tprio = 0 if agent_types.get(pc.rstrip("*"), "").lower() == "human" else 1
            return (cprio, tprio)
        return min(available, key=sort_key)

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE BUILDING
# ═══════════════════════════════════════════════════════════════════════════════

def build_table_columns(config):
    """Build DataTable column definitions with 3-level merged headers.

    Row 0 (top)  : "Activity Decomposition" | "Capacity Assessment" | "" (metadata)
    Row 1 (mid)  : ""                        | "Team Alternative N"  | ""
    Row 2 (bottom): actual column name
    """
    agent_set = set(get_agent_columns(config))

    # Map every agent column → its alternative label
    col_to_alt: dict[str, str] = {}
    for i, alt in enumerate(config.get("alternatives", []), 1):
        label = alt.get("name") or f"Team Alternative {i}"
        for col in alt.get("performers", []) + alt.get("supporters", []):
            col_to_alt[col] = label

    all_columns = config.get("all_columns", [])
    agent_indices = [i for i, c in enumerate(all_columns) if c in agent_set]
    first_agent_idx = min(agent_indices) if agent_indices else len(all_columns)

    # OPD columns (full names, abbreviations, single-letter) → "Teaming Requirements"
    TEAMING_COLS = {
        "Observability", "Predictability", "Directability",
        "Obs", "Pred", "Dir",
        "O", "P", "D",
    }
    # Role-assignment columns → "Teaming Structure"
    TEAMING_ROLE_COLS = {
        "TARS Performer Role", "TARS Supporter Role",
        "TA1 Performer Role", "TA1 supporter role",
        "TA2 performer role", "TA2 supporter role",
        "TA1 Supporter Role", "TA2 Performer Role",
    }

    columns = []
    for idx, col in enumerate(all_columns):
        if col in agent_set:
            alt_label = col_to_alt.get(col, "Team Alternative ?")
            name = ["Capacity Assessment", alt_label, col]
        elif idx < first_agent_idx:
            name = ["Activity Decomposition", " ", col]
        elif col in TEAMING_COLS:
            name = ["Teaming Requirements", "  ", col]
        elif col in TEAMING_ROLE_COLS:
            name = ["Teaming Structure", "   ", col]
        else:
            # Other metadata columns after the agent block (unique spacing
            # prevents accidental merging with neighbouring groups)
            name = ["  ", "    ", col]

        d = {"name": name, "id": col}
        if col == "Row":
            d["editable"] = False
        elif col in agent_set:
            d["editable"] = True
            d["presentation"] = "dropdown"
        else:
            d["editable"] = True
        # Allow users to hide/show metadata and teaming columns from the header
        if idx >= first_agent_idx and col not in agent_set:
            d["hideable"] = True
        columns.append(d)
    return columns


def build_dropdowns(config):
    return {
        col: {"options": [{"label": c.capitalize(), "value": c} for c in COLOR_OPTIONS]}
        for col in get_agent_columns(config)
    }


def build_borders(config):
    """Thick borders to visually separate team alternatives."""
    borders = []
    for alt in config.get("alternatives", []):
        all_in_alt = alt["performers"] + alt["supporters"]
        if all_in_alt:
            borders.append(
                {"if": {"column_id": all_in_alt[0]}, "borderLeft": "3px solid black"}
            )
            borders.append(
                {"if": {"column_id": all_in_alt[-1]}, "borderRight": "3px solid black"}
            )
    return borders


def style_table(df, config):
    """Conditional styles: color cells match their value.
    Uses filter_query so styles update live when the user edits a cell.
    """
    agent_cols = get_agent_columns(config)
    styles = []
    for col in agent_cols:
        for color in VALID_COLORS:
            for variant in [color, color.capitalize(), color.upper()]:
                styles.append({
                    "if": {
                        "filter_query": f"{{{col}}} = '{variant}'",
                        "column_id": col,
                    },
                    "backgroundColor": COLOR_MAP.get(color, color),
                    "color": COLOR_MAP.get(color, color),
                    "textAlign": "center",
                    "fontWeight": "bold",
                })
    return styles


def style_hierarchy_merge(df, config):
    """Visual pseudo-merge for all hierarchy columns (up to 4 levels).

    For each hierarchy level (except the last/task level):
    - Continuation rows where the value at this level (and all outer levels)
      is unchanged have their text made transparent, mimicking a merged cell.
    - The first row of a new group gets a top separator line whose thickness
      and darkness reflect the depth of the change: outermost level → thickest.
    """
    hierarchy_cols = config.get("hierarchy_columns", [])
    if not hierarchy_cols:
        # Backward compat: build from procedure_column + task_column
        proc_col = config.get("procedure_column", "Procedure")
        task_col = config.get("task_column", "Task")
        hierarchy_cols = [c for c in [proc_col, task_col] if c in df.columns]

    hierarchy_cols = [c for c in hierarchy_cols if c in df.columns]
    if len(hierarchy_cols) <= 1 or df.empty:
        return []

    all_cols = config.get("all_columns", [])
    styles = []
    df_reset = df.reset_index(drop=True)

    # Border width and colour for each hierarchy level (outermost = thickest/darkest)
    border_widths = [3, 2, 2, 1]
    border_colors = ["#333333", "#666666", "#999999", "#bbbbbb"]

    # prev_keys[level] = tuple of values at levels 0..level for the previous row
    prev_keys = [None] * len(hierarchy_cols)

    for i in range(len(df_reset)):
        curr_vals = [str(df_reset.at[i, c]).strip() for c in hierarchy_cols]
        curr_keys = [tuple(curr_vals[:lvl + 1]) for lvl in range(len(hierarchy_cols))]

        # Find the outermost (smallest index) level that changed
        change_level = None
        for lvl in range(len(hierarchy_cols)):
            if curr_keys[lvl] != prev_keys[lvl]:
                change_level = lvl
                break

        # Hide repeated text for display levels (all except the terminal/task level)
        for lvl in range(len(hierarchy_cols) - 1):
            col = hierarchy_cols[lvl]
            # Hide when: this level has not changed AND no outer level has changed
            if change_level is None or change_level > lvl:
                styles.append({
                    "if": {"row_index": i, "column_id": col},
                    "color": "transparent",
                    "borderTop": "1px solid #e8e8e8",
                })
            # If change_level <= lvl, an outer level changed → show this level's value

        # Draw a top separator when a non-terminal level changes (skip first row and
        # pure task-level changes, since those would add a border on every single row)
        if i > 0 and change_level is not None and change_level < len(hierarchy_cols) - 1:
            bw = border_widths[min(change_level, len(border_widths) - 1)]
            bc = border_colors[min(change_level, len(border_colors) - 1)]
            for col in all_cols:
                styles.append({
                    "if": {"row_index": i, "column_id": col},
                    "borderTop": f"{bw}px solid {bc}",
                })

        prev_keys = curr_keys

    return styles


# Keep old name as alias for backward compatibility
def style_procedure_merge(df, config):
    return style_hierarchy_merge(df, config)


# Columns always shown when present; everything else is hidden by default.
# Agent columns from the config are also always shown.
DEFAULT_VISIBLE_COLUMNS = {
    "Row", "Procedure", "Class", "Type", "Category", "Task", "Task Object",
    "Object", "Value",
    # Teaming requirements – full names, abbreviations, and single-letter forms
    "Observability", "Predictability", "Directability",
    "Obs", "Pred", "Dir",
    "O", "P", "D",
    # Teaming-structure / role-assignment columns
    "TARS Performer Role", "TARS Supporter Role",
    "TA1 Performer Role", "TA1 supporter role",
    "TA2 performer role", "TA2 supporter role",
    "TA1 Supporter Role", "TA2 Performer Role",  # capitalisation variants
}


def build_hidden_columns(config):
    """Return a list of column IDs that should be hidden by default."""
    agent_set = set(get_agent_columns(config))
    # Always show all hierarchy columns (they may not be in DEFAULT_VISIBLE_COLUMNS)
    hier_set = set(config.get("hierarchy_columns", [
        config.get("procedure_column", ""), config.get("task_column", "")
    ]))
    visible = DEFAULT_VISIBLE_COLUMNS | agent_set | hier_set
    return [c for c in config.get("all_columns", []) if c not in visible]


def build_data_table(df, config):
    """Create a new DataTable component from a DataFrame and config."""
    hidden = build_hidden_columns(config)
    return dash_table.DataTable(
        id="responsibility-table",
        columns=build_table_columns(config),
        data=df.to_dict("records"),
        editable=True,
        row_deletable=True,
        hidden_columns=hidden,
        merge_duplicate_headers=True,
        dropdown=build_dropdowns(config),
        style_data_conditional=style_table(df, config) + style_procedure_merge(df, config),
        style_cell={"textAlign": "left", "padding": "5px", "whiteSpace": "normal",
                    "fontFamily": "'Space Grotesk', 'Inter', sans-serif",
                    "backgroundColor": BG, "color": INK, "border": f"1px solid {BORDER}"},
        style_cell_conditional=build_borders(config),
        style_header={"fontWeight": "bold", "textAlign": "center"},
        style_header_conditional=[
            # Row 0 – top group labels
            {
                "if": {"header_index": 0},
                "backgroundColor": INK,
                "color": BG,
                "fontSize": "13px",
                "borderBottom": f"2px solid {BG}",
                "fontFamily": "'Space Grotesk', 'Inter', sans-serif",
                "letterSpacing": "0.04em",
                "textTransform": "uppercase",
            },
            # Row 1 – team alternative labels
            {
                "if": {"header_index": 1},
                "backgroundColor": "#3a3a3a",
                "color": BG,
                "fontSize": "12px",
                "borderBottom": f"2px solid {BG}",
                "fontFamily": "'Space Grotesk', 'Inter', sans-serif",
            },
            # Row 2 – column names (standard)
            {
                "if": {"header_index": 2},
                "backgroundColor": SURFACE,
                "color": INK,
                "fontSize": "12px",
                "fontFamily": "'Space Grotesk', 'Inter', sans-serif",
            },
        ],
        style_table={"overflowX": "auto", "border": f"2px solid {INK}"},
    )


def create_empty_df(config):
    """Create an empty DataFrame with one blank row for a new team."""
    all_cols = config.get("all_columns", ["Row", "Procedure", "Task"])
    row = {c: "" for c in all_cols}
    row["Row"] = 1
    return pd.DataFrame([row])


def ensure_row_column(df):
    """Make sure the Row column exists and is properly numbered."""
    df = df.copy()
    df["Row"] = range(1, len(df) + 1)
    return df


def config_summary_html(config):
    """Render team config as HTML for display."""
    if not config:
        return html.P("No configuration loaded.")

    agents_str = ", ".join(
        [f"{a['name']} ({a['type']})" for a in config["agents"]]
    )
    alt_items = []
    for alt in config["alternatives"]:
        perfs = ", ".join(alt["performers"])
        sups = ", ".join(alt["supporters"]) if alt["supporters"] else "None"
        alt_items.append(
            html.Li(f"{alt['name']}: Performers [{perfs}] — Supporters [{sups}]")
        )

    # Build column list with visible ones bolded
    agent_set = set(get_agent_columns(config))
    visible = DEFAULT_VISIBLE_COLUMNS | agent_set
    all_cols = config.get("all_columns", [])
    col_spans = []
    for i, col in enumerate(all_cols):
        label = html.B(col) if col in visible else html.Span(col, style={"color": "var(--ink-muted)"})
        col_spans.append(label)
        if i < len(all_cols) - 1:
            col_spans.append(", ")

    hier_cols = config.get("hierarchy_columns", [
        config.get("procedure_column", "Procedure"), config.get("task_column", "Task")
    ])
    level_names = ["Level 1 (broadest)", "Level 2", "Level 3", "Level 4 (task)"]
    hier_labels = [
        html.Li(f"{level_names[idx] if idx < len(level_names) else f'Level {idx+1}'}: {col}")
        for idx, col in enumerate(hier_cols)
    ]

    return html.Div([
        html.P([html.B("Agents: "), agents_str]),
        html.P(html.B("Hierarchy levels:")),
        html.Ul(hier_labels, style={"marginTop": "2px", "marginBottom": "8px"}),
        html.P([html.B("Category column: "), config.get("category_column") or "—  (none detected)"]),
        html.P([html.B("Columns: ")] + col_spans),
        html.Ul(alt_items),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# WORKFLOW GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

def build_workflow_figure_base(df, config, procedure=None, view_mode="full"):
    """
    Build the base workflow graph structure (without highlighting).

    Returns (fig, arrow_info) where arrow_info contains indices needed for
    the fast highlighting pass.
    """
    if config is None or df.empty:
        return go.Figure(), None

    agent_cols = get_agent_columns(config)
    performer_cols = set(get_performer_columns(config))
    task_col = config.get("task_column", "Task")

    # Performers-only view: filter to performer columns only
    if view_mode == "performers":
        agent_cols = [c for c in agent_cols if c in performer_cols]

    proc_col = config.get("procedure_column", "Procedure") if config else "Procedure"
    single_procedure = procedure is not None
    if single_procedure and proc_col in df.columns:
        df = df[df[proc_col] == procedure].copy()

    if df.empty:
        return go.Figure()

    df = df.reset_index(drop=True)
    df["task_idx"] = df.index
    tasks = df[task_col].tolist() if task_col in df.columns else [f"Task {i}" for i in range(len(df))]

    # ── Y-coordinate mapping ──────────────────────────────────────────────
    # When showing all procedures, insert a 1-unit gap between groups so
    # procedure-divider lines can be drawn in the extra space.
    # When filtered to a single procedure, y == task_idx (no gaps needed).
    GAP = 1.2          # extra y-units reserved for the divider between groups
    y_pos = []         # y_pos[i] = the plot y-coordinate for task i
    proc_dividers = [] # list of (y_between, proc_label) for separator lines
    procs = df[proc_col].tolist() if proc_col in df.columns else [""] * len(tasks)

    if single_procedure:
        y_pos = list(range(len(tasks)))
    else:
        y = 0.0
        # Seed the first procedure label above the first task
        if procs:
            proc_dividers.append((-0.6, procs[0]))
        for i, task_i in enumerate(tasks):
            if i > 0 and procs[i] != procs[i - 1]:
                # mid-point of the gap between groups
                proc_dividers.append((y + GAP / 2 - 0.5, procs[i]))
                y += GAP
            y_pos.append(y)
            y += 1.0

    agent_pos = {agent: i for i, agent in enumerate(agent_cols)}

    dots = []          # {task, y, agent, color}
    hover_lookup = {}  # (task_idx, col) -> hover text
    dashed_arrows = []

    # ── Per-task processing ───────────────────────────────────────────────
    for i, row in df.iterrows():
        task_idx = row["task_idx"]
        yp = y_pos[task_idx]
        task_label = wrap_text(str(row.get(task_col, "")))

        # Place dots + precompute hover text
        for col in agent_cols:
            if col in df.columns:
                val = str(row.get(col, "") or "").strip().lower()
                if val in VALID_COLORS:
                    dots.append({"task": task_idx, "y": yp, "agent": col, "color": val})
                    hover_parts = [
                        f"<b>Task:</b> {task_label}",
                        f"<b>Agent:</b> {col}",
                    ]
                    for meta in ["Observability", "Predictability", "Directability"]:
                        if meta in df.columns:
                            hover_parts.append(
                                f"<b>{meta}:</b><br>{wrap_text(str(row.get(meta, '')))}"
                            )
                    hover_lookup[(task_idx, col)] = "<br><br>".join(hover_parts)

        # Dashed arrows: supporter → performer within each alternative
        for alt in config["alternatives"]:
            active_perfs = [
                pc for pc in alt["performers"]
                if pc in df.columns
                and str(row.get(pc, "") or "").strip().lower() in VALID_COLORS
                and str(row.get(pc, "") or "").strip().lower() != "red"
            ]
            for sc in alt["supporters"]:
                if sc in df.columns:
                    sval = str(row.get(sc, "") or "").strip().lower()
                    if sval in VALID_COLORS and sval != "red":
                        for pc in active_perfs:
                            dashed_arrows.append({
                                "start_agent": sc,
                                "end_agent": pc,
                                "task": task_idx,
                                "y": yp,
                            })

    # ── Solid arrows between consecutive tasks ────────────────────────────
    performers_by_task = {}
    for d in dots:
        if d["agent"] in performer_cols and d["color"] != "red":
            performers_by_task.setdefault(d["task"], set()).add(d["agent"])

    solid_arrows = []
    for i in range(1, len(df)):
        for pa in performers_by_task.get(i - 1, set()):
            for ca in performers_by_task.get(i, set()):
                solid_arrows.append({
                    "start_task": i - 1, "start_y": y_pos[i - 1], "start_agent": pa,
                    "end_task": i,   "end_y":   y_pos[i],     "end_agent": ca,
                })

    # ── Render figure ─────────────────────────────────────────────────────
    fig = go.Figure()

    # Procedure divider lines + labels (rendered first so dots sit on top)
    x_min = -0.5
    x_max = len(agent_cols) - 0.5

    # Build alt-label lookup and per-alt column groups (only cols present in agent_cols)
    col_to_alt: dict[str, str] = {}
    alt_col_groups: list[tuple[str, list[int]]] = []  # (label, [x indices])
    for i, alt in enumerate(config.get("alternatives", []), 1):
        label = alt.get("name") or f"Alt {i}"
        cols_in_alt = [c for c in alt.get("performers", []) + alt.get("supporters", [])
                       if c in agent_pos]
        for col in cols_in_alt:
            col_to_alt[col] = label
        if cols_in_alt:
            alt_col_groups.append((label, [agent_pos[c] for c in cols_in_alt]))

    # In performers-only view anchor the procedure label to the left edge so it
    # doesn't sit on top of the connector lines that run through the centre.
    if view_mode == "performers":
        label_x = x_min
        label_xanchor = "left"
    else:
        label_x = (x_min + x_max) / 2
        label_xanchor = "center"
    for div_y, proc_label in proc_dividers:
        fig.add_shape(
            type="line",
            x0=x_min, y0=div_y, x1=x_max, y1=div_y,
            xref="x", yref="y",
            line=dict(color=INK_MUTED, width=1.5, dash="dot"),
        )
        fig.add_annotation(
            x=label_x, y=div_y,
            xref="x", yref="y",
            text=f"<b>{proc_label}</b>",
            showarrow=False,
            xanchor=label_xanchor,
            font=dict(size=11, color=INK),
            bgcolor="rgba(221,217,210,0.85)",
            bordercolor=BORDER,
            borderwidth=1,
            borderpad=4,
        )
        # One centred alt label per alternative group, sitting above the divider line
        for alt_label, x_indices in alt_col_groups:
            centre_x = sum(x_indices) / len(x_indices)
            fig.add_annotation(
                x=centre_x, y=div_y - 0.08,
                xref="x", yref="y",
                text=f"<i>{alt_label}</i>",
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                font=dict(size=9, color=INK_MUTED),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
            )
        # Per-column agent name annotations just above the divider line
        for col, x_idx in agent_pos.items():
            role = "Performer" if col in performer_cols else "Supporter"
            fig.add_annotation(
                x=x_idx, y=div_y - 0.22,
                xref="x", yref="y",
                text=f"<b>{col}</b><br><span style='font-size:8px'>{role}</span>",
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                font=dict(size=10, color=INK),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
            )

    # One scatter trace per agent column with per-point colors — much fewer traces
    for col in agent_cols:
        col_dots = [d for d in dots if d["agent"] == col]
        if not col_dots:
            continue
        fig.add_trace(go.Scatter(
            x=[agent_pos[col]] * len(col_dots),
            y=[d["y"] for d in col_dots],
            mode="markers",
            marker=dict(
                size=20,
                color=[COLOR_MAP.get(d["color"], d["color"]) for d in col_dots],
                symbol="circle",
            ),
            showlegend=False,
            hoverinfo="text",
            hovertext=[hover_lookup.get((d["task"], col), "") for d in col_dots],
        ))

    # Group dashed arrows by task for vertical offset
    dashed_by_task = {}
    for arrow in dashed_arrows:
        if arrow["start_agent"] in agent_pos and arrow["end_agent"] in agent_pos:
            dashed_by_task.setdefault(arrow["task"], []).append(arrow)

    dashed_arrow_info = []
    for task, arrows in dashed_by_task.items():
        n = len(arrows)
        offsets = [0] * n if n == 1 else [
            -0.08 + 0.16 * i / (n - 1) for i in range(n)
        ]
        for arrow, offset in zip(arrows, offsets):
            fig.add_shape(
                type="line",
                x0=agent_pos[arrow["start_agent"]], y0=arrow["y"] + offset,
                x1=agent_pos[arrow["end_agent"]], y1=arrow["y"] + offset,
                line=dict(color=INK, width=2, dash="dot"),
            )
            dashed_arrow_info.append({"task": arrow["task"], "end_agent": arrow["end_agent"]})

    solid_arrow_info = []
    for arrow in solid_arrows:
        fig.add_annotation(
            x=agent_pos[arrow["end_agent"]], y=arrow["end_y"],
            ax=agent_pos[arrow["start_agent"]], ay=arrow["start_y"],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1,
            arrowwidth=2,
            arrowcolor=INK,
            opacity=0.9,
        )
        solid_arrow_info.append({
            "start_task": arrow["start_task"], "start_agent": arrow["start_agent"],
            "end_task": arrow["end_task"], "end_agent": arrow["end_agent"],
        })

    # Layout
    title_suffix = f" — {procedure}" if single_procedure else " (All Procedures)"
    y_labels = [wrap_text(str(t)) for t in tasks]

    # Estimated height: task rows + gap rows
    total_y_span = y_pos[-1] if y_pos else len(tasks)
    height = max(400, 100 + int(total_y_span * 80))

    fig.update_layout(
        title=f"Workflow Graph{title_suffix}",
        xaxis=dict(
            tickvals=list(agent_pos.values()),
            ticktext=list(agent_pos.keys()),
            title="Agent",
            showgrid=True, gridcolor=BORDER,
            range=[x_min, x_max],
        ),
        yaxis=dict(
            tickvals=y_pos,
            ticktext=y_labels,
            title="Task",
            range=[(y_pos[-1] + 0.5) if y_pos else len(tasks), -1.0],
            showgrid=False,
        ),
        height=height,
        margin=dict(l=250, r=50, t=50, b=50),
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font=dict(family="Space Grotesk, Inter, sans-serif", color=INK),
    )

    arrow_info = {
        "n_divider_shapes": len(proc_dividers),
        # 1 proc-label + 1 per alt group + 1 per agent column, all per divider
        "n_divider_annotations": len(proc_dividers) * (1 + len(alt_col_groups) + len(agent_cols)),
        "dashed_arrows": dashed_arrow_info,
        "solid_arrows": solid_arrow_info,
    }
    return fig, arrow_info


def apply_workflow_highlighting(fig_dict, arrow_info, df, config, procedure=None,
                                highlight_track=None, category_overrides=None):
    """Apply highlighting to a cached base workflow figure. Fast: only updates colors/widths."""
    if fig_dict is None:
        return go.Figure()

    fig = go.Figure(fig_dict)

    if arrow_info is None:
        return fig

    if (not highlight_track or highlight_track == "none") and not category_overrides:
        return fig

    if category_overrides is None:
        category_overrides = {}

    proc_col = config.get("procedure_column", "Procedure") if config else "Procedure"

    if procedure is not None and proc_col in df.columns:
        df = df[df[proc_col] == procedure].copy()

    df = df.reset_index(drop=True)
    df["task_idx"] = df.index

    should_hl_support = highlight_track in (
        "human_full_support", "agent_whenever_possible_full_support", "most_reliable",
    )

    # Build highlight set
    highlight_set = set()
    if highlight_track and highlight_track != "none":
        for _, row in df.iterrows():
            chosen = get_chosen_performer(row, config, highlight_track, category_overrides)
            if chosen:
                highlight_set.add((row["task_idx"], chosen))

    # Apply highlighting to dashed arrow shapes
    if fig.layout.shapes:
        shapes = list(fig.layout.shapes)
        offset = arrow_info.get("n_divider_shapes", 0)
        for i, info in enumerate(arrow_info.get("dashed_arrows", [])):
            idx = offset + i
            if idx < len(shapes):
                is_hl = should_hl_support and (info["task"], info["end_agent"]) in highlight_set
                shapes[idx].line.color = ACCENT if is_hl else INK
                shapes[idx].line.width = 4 if is_hl else 2
        fig.layout.shapes = shapes

    # Apply highlighting to solid arrow annotations
    if fig.layout.annotations:
        annotations = list(fig.layout.annotations)
        offset = arrow_info.get("n_divider_annotations", 0)
        for i, info in enumerate(arrow_info.get("solid_arrows", [])):
            idx = offset + i
            if idx < len(annotations):
                is_hl = bool(highlight_set) and \
                        (info["start_task"], info["start_agent"]) in highlight_set and \
                        (info["end_task"], info["end_agent"]) in highlight_set
                annotations[idx].arrowcolor = ACCENT if is_hl else INK
                annotations[idx].arrowwidth = 4 if is_hl else 2
        fig.layout.annotations = annotations

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# BAR CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def build_capacity_bar_chart(df, config):
    """Bar chart showing performer/supporter capacities per agent."""
    if config is None or df.empty:
        return go.Figure()

    performer_cols = get_performer_columns(config)
    supporter_cols = get_supporter_columns(config)
    color_shades = COLOR_SHADES

    fig = go.Figure()

    for grade in ["green", "yellow", "orange"]:
        shades = color_shades[grade]
        # Performers
        for i, col in enumerate(performer_cols):
            base_name = col.rstrip("*")
            count = sum(
                1 for _, row in df.iterrows()
                if str(row.get(col, "") or "").strip().lower() == grade
            )
            fig.add_trace(go.Bar(
                name=base_name,
                x=[f"Performer {grade.capitalize()}"],
                y=[count],
                marker_color=shades[i % len(shades)],
                showlegend=False,
                text=[base_name], textposition="outside", textangle=0,
            ))
        # Supporters
        for i, col in enumerate(supporter_cols):
            count = sum(
                1 for _, row in df.iterrows()
                if str(row.get(col, "") or "").strip().lower() == grade
            )
            fig.add_trace(go.Bar(
                name=col,
                x=[f"Supporter {grade.capitalize()}"],
                y=[count],
                marker_color=shades[i % len(shades)],
                showlegend=False,
                text=[col], textposition="outside", textangle=0,
            ))

    fig.update_layout(
        title="Performer and Supporter Capacities",
        xaxis_title="Role and Capacity",
        yaxis_title="Number of Tasks",
        barmode="group", bargap=0.15, bargroupgap=0.1,
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(family="Space Grotesk, Inter, sans-serif", color=INK),
        showlegend=False,
    )
    return fig


def build_allocation_bar_chart(df, config):
    """Pie chart showing task type (allocation) distribution."""
    if config is None or df.empty:
        return go.Figure()

    performer_cols = get_performer_columns(config)
    supporter_cols = get_supporter_columns(config)

    single_independent = 0
    multiple_independent = 0
    single_interdependent = 0
    multiple_interdependent = 0

    for _, row in df.iterrows():
        perfs = [
            c for c in performer_cols
            if c in df.columns
            and str(row.get(c, "") or "").strip().lower() in VALID_COLORS
            and str(row.get(c, "") or "").strip().lower() != "red"
        ]
        sups = [
            c for c in supporter_cols
            if c in df.columns
            and str(row.get(c, "") or "").strip().lower() in VALID_COLORS
            and str(row.get(c, "") or "").strip().lower() != "red"
        ]
        if len(sups) > 0:
            if len(perfs) == 1:
                single_interdependent += 1
            else:
                multiple_interdependent += 1
        elif len(perfs) == 1:
            single_independent += 1
        elif len(perfs) > 1:
            multiple_independent += 1

    labels = [
        "Single Allocation Independent",
        "Multiple Allocation Independent",
        "Single Allocation Interdependent",
        "Multiple Allocation Interdependent",
    ]
    values = [single_independent, multiple_independent, single_interdependent, multiple_interdependent]

    # High-contrast fills:
    #  1. solid white
    #  2. white + faint grey dots
    #  3. white + bold ink diagonal lines
    #  4. dark grey solid (no pattern needed)
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        marker=dict(
            colors=["white", "white", "white", DARK_GREY],
            pattern=dict(
                shape=["", ".", "/", ""],
                fgcolor=[INK, BORDER, INK, DARK_GREY],
                size=[6, 6, 7, 6],
                solidity=[1.0, 0.35, 0.75, 1.0],
            ),
            line=dict(color=INK, width=2),
        ),
        textinfo="label+percent",
        textposition="outside",
        hovertemplate="%{label}<br>Count: %{value}<br>%{percent}<extra></extra>",
        hole=0.3,
        showlegend=False,
    ))
    fig.update_layout(
        title="Task Type Distribution",
        height=520,
        paper_bgcolor=BG,
        font=dict(family="Space Grotesk, Inter, sans-serif", color=INK),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def build_autonomy_bar_chart(df, config):
    """Pie charts showing agent autonomy (task continuity) — one pie per performer."""
    if config is None or df.empty:
        return go.Figure()

    from plotly.subplots import make_subplots

    performer_cols = get_performer_columns(config)
    agent_autonomy = {c: {"autonomous": 0, "non_autonomous": 0} for c in performer_cols}

    prev_performers = set()
    agent_seen = {c: False for c in performer_cols}
    for _, row in df.iterrows():
        current_performers = set()
        for col in performer_cols:
            if col in df.columns:
                val = str(row.get(col, "") or "").strip().lower()
                if val in VALID_COLORS and val != "red":
                    current_performers.add(col)
                    if val == "orange":
                        agent_autonomy[col]["non_autonomous"] += 1
                    elif not agent_seen[col]:
                        pass  # first task for this agent — no prior task to compare, skip
                    elif col not in prev_performers:
                        agent_autonomy[col]["non_autonomous"] += 1
                    else:
                        agent_autonomy[col]["autonomous"] += 1
                    agent_seen[col] = True
        prev_performers = current_performers

    active_cols = [c for c in performer_cols if (agent_autonomy[c]["autonomous"] + agent_autonomy[c]["non_autonomous"]) > 0]
    if not active_cols:
        return go.Figure()

    n = len(active_cols)
    fig = make_subplots(
        rows=1, cols=n,
        specs=[[{"type": "pie"}] * n],
        subplot_titles=[c.rstrip("*") for c in active_cols],
    )

    for i, col in enumerate(active_cols, 1):
        auto = agent_autonomy[col]["autonomous"]
        non_auto = agent_autonomy[col]["non_autonomous"]
        fig.add_trace(go.Pie(
            labels=["Autonomous", "Non-Autonomous"],
            values=[auto, non_auto],
            marker=dict(
                colors=["white", "#555250"],   # solid white / dark grey solid
                pattern=dict(
                    shape=["", ""],
                    fgcolor=[INK, "#555250"],
                    size=[6, 6],
                    solidity=1.0,
                ),
                line=dict(color=INK, width=2),
            ),
            textinfo="label+percent",
            textposition="outside",
            hovertemplate="%{label}<br>Count: %{value}<br>%{percent}<extra></extra>",
            hole=0.3,
            showlegend=False,
        ), row=1, col=i)

    fig.update_layout(
        title="Agent Autonomy: Task Continuity",
        height=400,
        paper_bgcolor=BG,
        font=dict(family="Space Grotesk, Inter, sans-serif", color=INK),
        margin=dict(l=20, r=20, t=80, b=20),
    )
    return fig


def build_most_reliable_bar_chart(df, config):
    """Performer/supporter capacities along the most reliable path."""
    if config is None or df.empty:
        return go.Figure()

    perf_green, perf_yellow, perf_orange = 0, 0, 0
    sup_green, sup_yellow, sup_orange = 0, 0, 0

    for _, row in df.iterrows():
        chosen = get_chosen_performer(row, config, "most_reliable")
        if not chosen:
            continue
        val = str(row.get(chosen, "") or "").strip().lower()
        if val == "green":
            perf_green += 1
        elif val == "yellow":
            perf_yellow += 1
        elif val == "orange":
            perf_orange += 1

        # Find active supporter for the chosen performer's alternative
        for alt in config["alternatives"]:
            if chosen in alt["performers"]:
                for sc in alt["supporters"]:
                    if sc in df.columns:
                        sval = str(row.get(sc, "") or "").strip().lower()
                        if sval == "green":
                            sup_green += 1
                        elif sval == "yellow":
                            sup_yellow += 1
                        elif sval == "orange":
                            sup_orange += 1

    fig = go.Figure()
    for label, count, color in [
        ("Performer Green", perf_green, PAL_GREEN),
        ("Performer Yellow", perf_yellow, PAL_YELLOW),
        ("Performer Orange", perf_orange, PAL_ORANGE),
        ("Supporter Green", sup_green, COLOR_MAP_LIGHT["green"]),
        ("Supporter Yellow", sup_yellow, COLOR_MAP_LIGHT["yellow"]),
        ("Supporter Orange", sup_orange, COLOR_MAP_LIGHT["orange"]),
    ]:
        fig.add_trace(go.Bar(name=label, x=[label], y=[count], marker_color=color, showlegend=False))

    fig.update_layout(
        title="Most Reliable Path: Performer and Supporter Capacities",
        xaxis_title="Role and Capacity", yaxis_title="Number of Tasks",
        barmode="group", bargap=0.15,
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(family="Space Grotesk, Inter, sans-serif", color=INK),
        showlegend=False,
    )
    return fig


def build_human_baseline_bar_chart(df, config):
    """Human-only performer capacities (no support, no autonomous agents)."""
    if config is None or df.empty:
        return go.Figure()

    agent_types = {a["name"]: a["type"] for a in config["agents"]}
    performer_cols = get_performer_columns(config)
    human_perfs = [
        pc for pc in performer_cols
        if agent_types.get(pc.rstrip("*"), "").lower() == "human"
    ]
    if not human_perfs:
        return go.Figure()

    perf_green, perf_yellow, perf_orange = 0, 0, 0
    for _, row in df.iterrows():
        for pc in human_perfs:
            if pc in df.columns:
                val = str(row.get(pc, "") or "").strip().lower()
                if val == "green":
                    perf_green += 1
                elif val == "yellow":
                    perf_yellow += 1
                elif val == "orange":
                    perf_orange += 1

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Green", x=["Green"], y=[perf_green], marker_color=PAL_GREEN, showlegend=False))
    fig.add_trace(go.Bar(name="Yellow", x=["Yellow"], y=[perf_yellow], marker_color=PAL_YELLOW, showlegend=False))
    fig.add_trace(go.Bar(name="Orange", x=["Orange"], y=[perf_orange], marker_color=PAL_ORANGE, showlegend=False))

    fig.update_layout(
        title="Human-Only Baseline: Human Performer Capacities",
        xaxis_title="Capacity Level", yaxis_title="Number of Tasks",
        barmode="group", bargap=0.15,
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(family="Space Grotesk, Inter, sans-serif", color=INK),
        showlegend=False,
    )
    return fig


# ─── Automation Proportion ────────────────────────────────────────────────────

def compute_automation_proportion_data(df, config, highlight_track, category_overrides=None):
    """
    Compute automation proportion range [P_min, P_max] using Liu & Kaber (2025) method.

    Support is opportunistic (optional) unless the performer value is orange (mandatory).
    For agent performers, human support is optional unless the human supporter value is orange.

    Per-task weight ranges:
      Human performer, orange   → mandatory support     → w = 0.5  (fixed)
      Human performer, grn/yel  → optional auto-support → w ∈ [0.0, 0.5]
      Human performer, no sup   →                        w = 0.0  (fixed)
      Agent performer, orange human supporter → mandatory → w = 0.75 (fixed)
      Agent performer, grn/yel human supporter → optional → w ∈ [0.75, 1.0]
      Agent performer, no human support        →             w = 1.0  (fixed)

    Returns (P_min, P_max, category_scores_dict) where each entry is (pk_min, pk_max),
    or (None, None, {}) if not applicable.
    """
    cat_col = config.get("category_column") or "Category"
    if config is None or df.empty or cat_col not in df.columns:
        return None, None, {}

    if not highlight_track or highlight_track == "none":
        return None, None, {}

    agent_types = {a["name"]: a["type"] for a in config["agents"]}
    categories = sorted(df[cat_col].dropna().unique())

    if not categories:
        return None, None, {}

    K = len(categories)
    category_scores = {}
    is_full_support = highlight_track in (
        "human_full_support", "agent_whenever_possible_full_support", "most_reliable",
    )

    for cat in categories:
        cat_df = df[df[cat_col] == cat]
        Tk = len(cat_df)
        if Tk == 0:
            continue

        w_mins, w_maxs = [], []
        for _, row in cat_df.iterrows():
            chosen = get_chosen_performer(row, config, highlight_track, category_overrides)

            if chosen is None:
                w_mins.append(0.0)
                w_maxs.append(0.0)
                continue

            chosen_type = agent_types.get(chosen.rstrip("*"), "autonomous")
            chosen_val = str(row.get(chosen, "") or "").strip().lower()

            if chosen_type == "autonomous":
                if not is_full_support:
                    w_mins.append(1.0)
                    w_maxs.append(1.0)
                else:
                    # Look for human supporters in the same alternative
                    has_mandatory = False
                    has_optional = False
                    for alt in config["alternatives"]:
                        if chosen in alt["performers"]:
                            for sup_col in alt["supporters"]:
                                sup_type = agent_types.get(sup_col.rstrip("*"), "autonomous")
                                if sup_type == "human" and sup_col in df.columns:
                                    sval = str(row.get(sup_col, "") or "").strip().lower()
                                    if sval in VALID_COLORS and sval != "red":
                                        if sval == "orange":
                                            has_mandatory = True
                                        else:
                                            has_optional = True
                    if has_mandatory:
                        w_mins.append(0.75)
                        w_maxs.append(0.75)
                    elif has_optional:
                        w_mins.append(0.75)
                        w_maxs.append(1.0)
                    else:
                        w_mins.append(1.0)
                        w_maxs.append(1.0)

            else:  # human performer
                if not is_full_support:
                    w_mins.append(0.0)
                    w_maxs.append(0.0)
                else:
                    if chosen_val == "orange":
                        # Mandatory support
                        w_mins.append(0.5)
                        w_maxs.append(0.5)
                    else:
                        # Check for autonomous supporters
                        has_auto_support = False
                        for alt in config["alternatives"]:
                            if chosen in alt["performers"]:
                                for sup_col in alt["supporters"]:
                                    sup_type = agent_types.get(sup_col.rstrip("*"), "autonomous")
                                    if sup_type != "human" and sup_col in df.columns:
                                        sval = str(row.get(sup_col, "") or "").strip().lower()
                                        if sval in VALID_COLORS and sval != "red":
                                            has_auto_support = True
                        if has_auto_support:
                            w_mins.append(0.0)
                            w_maxs.append(0.5)
                        else:
                            w_mins.append(0.0)
                            w_maxs.append(0.0)

        pk_min = sum(w_mins) / Tk
        pk_max = sum(w_maxs) / Tk
        category_scores[cat] = (pk_min, pk_max)

    P_min = sum(v[0] for v in category_scores.values()) / K if K > 0 else 0.0
    P_max = sum(v[1] for v in category_scores.values()) / K if K > 0 else 0.0
    return P_min, P_max, category_scores


# ═══════════════════════════════════════════════════════════════════════════════
# APP LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

SHOW = {}
HIDE = {"display": "none"}

app.layout = html.Div([
    # ── Stores ──
    dcc.Store(id="team-config-store", data=None),
    dcc.Store(id="category-overrides-store", data={}),
    dcc.Store(id="base-figure-store", data=None),
    dcc.Store(id="arrow-indices-store", data=None),
    dcc.Store(id="pending-csv-store", data=None),

    # ═══════════════════════════════════════════════════════════════════════
    # STICKY NAVIGATION HEADER
    # ═══════════════════════════════════════════════════════════════════════
    html.Nav(className="sticky-nav", children=[
        html.A("Interdependence Analysis Dashboard", href="#", className="nav-brand"),
        html.Div(id="nav-links", className="nav-links", style=HIDE, children=[
            html.A("Table",      href="#table-anchor",      className="nav-link"),
            html.A("Workflow",   href="#workflow-anchor",   className="nav-link"),
            html.A("Statistics", href="#statistics-anchor", className="nav-link"),
        ]),
    ]),

    # ═══════════════════════════════════════════════════════════════════════
    # SETUP SECTION  [01]
    # ═══════════════════════════════════════════════════════════════════════
    html.Div(id="setup-section", className="section", children=[
        html.Div("[ 01 ]", className="section-number"),
        html.H2("Team Setup", className="section-title"),
        html.P(
            "Upload a CSV or manually define your Human-Autonomy Team configuration.",
            className="section-subtitle",
        ),

        # ── Option 1: CSV Upload ──
        html.Div(className="card", children=[
            html.Div("Option 1 — Load from CSV", className="card-header"),
            html.P(
                "Upload a CSV and the team structure will be auto-detected from "
                "columns containing color values (red/yellow/green/orange). "
                "Columns ending with * are treated as performer roles.",
                style={"fontSize": "14px"},
            ),
            html.P([
                html.B("Expected column structure: "),
                "Row | Procedure | Task | [optional: Category] | Agent columns… | Metadata columns",
            ], style={"fontSize": "13px", "fontFamily": "var(--font-mono)"}),
            html.P([
                "The Procedure column groups Tasks hierarchically (Procedure → Task). "
                "The optional Category column (e.g. Observe/Orient/Decide/Act from a OODA decomposition) "
                "is used for Automation Proportion computation. Any column with few unique string values "
                "not matching colors will be auto-detected as the Category column.",
            ], style={"fontSize": "13px"}),
            dcc.Upload(
                id="setup-upload",
                children=html.Div([
                    "Drag and Drop or ",
                    html.A("Select a CSV File", style={"color": ACCENT, "cursor": "pointer", "fontWeight": "600"}),
                ]),
                className="upload-zone",
                style={
                    "width": "100%", "lineHeight": "60px",
                    "textAlign": "center", "margin": "10px 0",
                },
                multiple=False,
            ),
            html.Div([
                html.Button(
                    "Load Example (IA_V8.csv)",
                    id="load-example-button",
                    n_clicks=0,
                    style={"marginTop": "8px", "fontSize": "13px"},
                ),
                html.Span(
                    " — load a pre-built example to explore the dashboard",
                    style={"fontSize": "12px", "color": INK_MUTED, "marginLeft": "8px"},
                ),
            ]),
            html.Div(id="upload-status", style={"marginTop": "10px", "fontStyle": "italic"}),

            # ── Hierarchy configuration panel (shown after CSV upload) ──
            html.Div(id="hierarchy-config-section", style=HIDE, children=[
                html.H4("Configure Hierarchy Levels",
                        style={"marginTop": "18px", "textTransform": "uppercase",
                               "letterSpacing": "0.04em", "fontSize": "13px", "color": INK_MUTED}),
                html.P(
                    "Select 1–4 columns that form the hierarchical decomposition of tasks, "
                    "from the broadest grouping (Level 1) down to the task level (last active level). "
                    "Auto-detected values are pre-filled — adjust if needed.",
                    style={"fontSize": "13px"},
                ),
                html.Div([
                    html.Div([
                        html.Label("Level 1 (broadest):",
                                   style={"fontWeight": "bold", "fontSize": "12px", "display": "block"}),
                        dcc.Dropdown(id="hier-level-1", options=[], value=None, clearable=True,
                                     placeholder="e.g. Phase, Procedure",
                                     style={"fontSize": "13px"}),
                    ], style={"flex": "1", "marginRight": "8px"}),
                    html.Div([
                        html.Label("Level 2:",
                                   style={"fontWeight": "bold", "fontSize": "12px", "display": "block"}),
                        dcc.Dropdown(id="hier-level-2", options=[], value=None, clearable=True,
                                     placeholder="e.g. Goal",
                                     style={"fontSize": "13px"}),
                    ], style={"flex": "1", "marginRight": "8px"}),
                    html.Div([
                        html.Label("Level 3:",
                                   style={"fontWeight": "bold", "fontSize": "12px", "display": "block"}),
                        dcc.Dropdown(id="hier-level-3", options=[], value=None, clearable=True,
                                     placeholder="e.g. Subgoal",
                                     style={"fontSize": "13px"}),
                    ], style={"flex": "1", "marginRight": "8px"}),
                    html.Div([
                        html.Label("Level 4 (task level):",
                                   style={"fontWeight": "bold", "fontSize": "12px", "display": "block"}),
                        dcc.Dropdown(id="hier-level-4", options=[], value=None, clearable=True,
                                     placeholder="e.g. Required capacity",
                                     style={"fontSize": "13px"}),
                    ], style={"flex": "1"}),
                ], style={"display": "flex", "gap": "8px", "marginBottom": "14px"}),
                html.Button(
                    "Load with this hierarchy",
                    id="confirm-hierarchy-button",
                    n_clicks=0,
                    className="btn-primary",
                ),
            ]),
        ]),

        # ── Option 2: Manual Setup ──
        html.Div(className="card-alt", children=[
            html.Div("Option 2 — Define Team Manually", className="card-header"),
            html.P(
                "The first task of the IA requires defining team alternatives, first list all of the agents "
                "in the team, then check for alternatives in agent's roles (e.g. performer vs supporter). "
                "Enter the agent role columns in order. Use * for performer columns. "
                "Group them as: Alt1-performers, Alt1-supporters, Alt2-performers, Alt2-supporters, …",
                style={"fontSize": "14px"},
            ),
            html.Div([
                html.Label("Agent columns (comma-separated):", style={"fontWeight": "bold"}),
                dcc.Input(
                    id="manual-columns-input",
                    value="Human*, Robot, Robot*, Human",
                    style={"width": "100%", "marginBottom": "10px", "padding": "8px"},
                ),
            ]),

            # ── Procedure column ──
            html.Div([
                html.Label("Higher-level activity column name:", style={"fontWeight": "bold", "marginRight": "10px"}),
                dcc.Input(
                    id="manual-procedure-col-input",
                    value="Procedure",
                    style={"width": "200px", "padding": "8px"},
                ),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
            html.P(
                "Define the column name that groups several Tasks and represents one level of hierarchical "
                "decomposition of the joint activity — the analysis uses a two-level hierarchy: "
                "(e.g. Procedure → Task.) This column typically corresponds to high-level mission phases "
                "or activity clusters.",
                style={"fontSize": "13px", "marginTop": "2px", "marginBottom": "14px"},
            ),

            # ── Task column ──
            html.Div([
                html.Label("Lower-level activity column name:", style={"fontWeight": "bold", "marginRight": "10px"}),
                dcc.Input(
                    id="manual-task-col-input",
                    value="Task",
                    style={"width": "200px", "padding": "8px"},
                ),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
            html.P([
                "Represents the terminal nodes of the activity decomposition. We recommend decomposing into "
                "required capacity in terms of information-processing stages: ",
                html.B("Sense → Interpret → Decide → Act"),
            ], style={"fontSize": "13px", "marginTop": "2px", "marginBottom": "14px"}),

            # ── Category column ──
            html.Div([
                html.Label("Category column name (optional):", style={"fontWeight": "bold", "marginRight": "10px"}),
                dcc.Input(
                    id="manual-category-col-input",
                    value="Category",
                    placeholder="e.g. Category, Type, Stage",
                    style={"width": "220px", "padding": "8px"},
                ),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),
            html.P(
                "The Category column assigns each task to a named group. It is used as the grouping variable "
                "for Automation Proportion computation: one proportion pₖ is computed per category, then "
                "averaged to produce the overall index P. Leave blank if not applicable.",
                style={"fontSize": "13px", "marginTop": "2px", "marginBottom": "14px"},
            ),

            # Preset buttons
            html.Div([
                html.Label("Presets: ", style={"fontWeight": "bold", "marginRight": "10px"}),
                html.Button("Human + Robot", id="preset-2agent", n_clicks=0,
                            style={"marginRight": "10px"}),
                html.Button("Human + UGV + UAV", id="preset-3agent", n_clicks=0,
                            style={"marginRight": "10px"}),
            ], style={"marginBottom": "15px"}),
            html.Button(
                "Create Team", id="create-team-button", n_clicks=0,
                className="btn-primary",
                style={"marginTop": "10px"},
            ),
        ]),
        # ── References ──
        html.Div(style={"marginTop": "3rem", "borderTop": f"1px solid {BORDER}", "paddingTop": "1.5rem"}, children=[
            html.P([
                "If you want to know more about interdependence analysis for building effective Human-Autonomy Teams, "
                "check these publications and these two videos by Matthew Johnson: ",
                html.A("Video 1", href="https://www.youtube.com/watch?v=BnuTBMWnf6M",
                       target="_blank", style={"color": ACCENT, "fontWeight": "600"}),
                " · ",
                html.A("Video 2", href="https://www.youtube.com/watch?v=M2UgTNPjHyM",
                       target="_blank", style={"color": ACCENT, "fontWeight": "600"}),
                ".",
            ], style={"fontSize": "14px", "marginBottom": "1.2rem", "lineHeight": "1.7"}),
            html.H4("References", style={"textTransform": "uppercase", "letterSpacing": "0.05em",
                                         "fontSize": "13px", "color": INK_MUTED, "marginBottom": "1rem"}),
            html.Ol(style={"fontSize": "13px", "lineHeight": "1.8", "paddingLeft": "1.2rem",
                           "color": INK, "fontFamily": "var(--font-body)"}, children=[
                html.Li("Johnson, M. (2014). Coactive Design: Designing Support for Interdependence in Human-Robot Teamwork."),
                html.Li([
                    "Johnson, M., Bradshaw, J., & Feltovich, P. J. (2017). Tomorrow's Human–Machine Design Tools: From Levels of Automation to Interdependencies. ",
                    html.Em("Journal of Cognitive Engineering and Decision Making"), ", 12, 155534341773646. ",
                    html.A("https://doi.org/10.1177/1555343417736462",
                           href="https://doi.org/10.1177/1555343417736462", target="_blank",
                           style={"color": ACCENT}),
                ]),
                html.Li([
                    "Johnson, M., Bradshaw, J., Feltovich, P. J., Hoffman, R., Jonker, C., Riemsdijk, B., & Sierhuis, M. (2011). Beyond Cooperative Robotics: The Central Role of Interdependence in Coactive Design. ",
                    html.Em("IEEE Intelligent Systems"), ", 26, 81–88. ",
                    html.A("https://doi.org/10.1109/MIS.2011.47",
                           href="https://doi.org/10.1109/MIS.2011.47", target="_blank",
                           style={"color": ACCENT}),
                ]),
                html.Li([
                    "Johnson, M., & Bradshaw, J. M. (2021). How Interdependence Explains the World of Teamwork. In W. F. Lawless, J. Llinas, D. A. Sofge, & R. Mittu (Eds.), ",
                    html.Em("Engineering Artificially Intelligent Systems: A Systems Engineering Approach to Realizing Synergistic Capabilities"),
                    " (pp. 122–146). Springer International Publishing. ",
                    html.A("https://doi.org/10.1007/978-3-030-89385-9_8",
                           href="https://doi.org/10.1007/978-3-030-89385-9_8", target="_blank",
                           style={"color": ACCENT}),
                ]),
                html.Li([
                    "Johnson, M., Bradshaw, J. M., Feltovich, P. J., Jonker, C. M., van Riemsdijk, M. B., & Sierhuis, M. (2014). Coactive design: Designing support for interdependence in joint activity. ",
                    html.Em("J. Hum.-Robot Interact."), ", 3(1), 43–69. ",
                    html.A("https://doi.org/10.5898/JHRI.3.1.Johnson",
                           href="https://doi.org/10.5898/JHRI.3.1.Johnson", target="_blank",
                           style={"color": ACCENT}),
                ]),
                html.Li([
                    "Johnson, M., Vignati, M., & Duran, D. (2018). Understanding Human-Autonomy Teaming through Interdependence Analysis. ",
                    html.A("ihmc",
                           href="https://www.ihmc.us/wp-content/uploads/2019/01/180907-HAT-Interdependence-Analysis.pdf",
                           target="_blank", style={"color": ACCENT}),
                ]),
            ]),
        ]),
    ]),

    # ═══════════════════════════════════════════════════════════════════════
    # CONFIG SUMMARY (shown after setup)
    # ═══════════════════════════════════════════════════════════════════════
    html.Div(id="config-summary", style=HIDE, children=[
        html.Div(className="section", style={"paddingBottom": "1rem"}, children=[
            html.Div("[ 01 ]", className="section-number"),
            html.Div([
                html.H4("Current Team Configuration", style={"display": "inline-block", "margin": "0"}),
                html.Button("Change Team", id="reset-config-button", n_clicks=0,
                            style={"marginLeft": "20px"}),
            ], style={"display": "flex", "alignItems": "center"}),
            html.Div(id="config-details", style={"marginTop": "1rem"}),
        ]),
    ]),

    # ═══════════════════════════════════════════════════════════════════════
    # ANALYSIS SECTION
    # ═══════════════════════════════════════════════════════════════════════
    html.Div(id="analysis-section", style=HIDE, children=[

        # ── [02] Table ──
        html.Div(id="table-anchor", className="section", children=[
            html.Div("[ 02 ]", className="section-number"),
            html.H2("Interdependence Analysis Table", className="section-title"),

            html.Div(id="table-wrapper"),

            # Action buttons
            html.Div(style={"marginTop": "1rem"}, children=[
                html.Div([
                    html.Div([
                        dcc.Upload(
                            id="upload-data",
                            children=html.Button("Load Table", id="load-button", n_clicks=0),
                            multiple=False,
                            style={"display": "inline-block", "marginRight": "10px"},
                        ),
                        html.Button("Add Row", id="add-row-button", n_clicks=0),
                        html.Button("Copy Cell Down", id="copy-down-button", n_clicks=0),
                    ], style={"display": "flex", "gap": "10px"}),
                    html.Div([
                        html.Button("Save Table", id="save-button", n_clicks=0),
                        dcc.Download(id="download-csv"),
                    ], style={"marginLeft": "auto"}),
                ], style={"display": "flex", "width": "100%"}),
                # Export row
                html.Div([
                    html.Button("Copy as Rich Text", id="copy-markdown-button", n_clicks=0),
                    html.Button("Export PNG", id="export-table-png-button", n_clicks=0),
                    dcc.Download(id="download-table-png"),
                    html.Span(id="copy-markdown-status",
                              style={"marginLeft": "12px", "fontStyle": "italic", "fontSize": "13px"}),
                ], style={"display": "flex", "gap": "10px", "alignItems": "center", "marginTop": "8px"}),
                html.Div(id="save-confirmation", style={"marginTop": "10px", "fontStyle": "italic"}),
            ]),
        ]),

        # ── [03] Workflow Graph ──
        html.Div(id="workflow-anchor", className="section", children=[
            html.Div("[ 03 ]", className="section-number"),
            html.H2("Workflow Graph", className="section-title"),

            # Procedure dropdown + View selector
            html.Div([
                dcc.Dropdown(
                    id="procedure-dropdown",
                    options=[],
                    value=None,
                    placeholder="Select a procedure to filter the graph…",
                    clearable=True,
                    style={"width": "320px", "marginRight": "30px"},
                ),
                dcc.RadioItems(
                    id="view-selector",
                    options=[
                        {"label": "Full View (All Agents)", "value": "full"},
                        {"label": "Performers Only", "value": "performers"},
                    ],
                    value="full",
                    labelStyle={"display": "inline-block", "marginRight": "20px"},
                    style={"display": "flex", "alignItems": "center"},
                ),
            ], style={"display": "flex", "alignItems": "center", "marginTop": "12px"}),

            # Allocation pattern selector
            dcc.RadioItems(
                id="highlight-selector",
                options=[
                    {"label": "No highlight", "value": "none"},
                    {"label": "Alt 1 performer — independent", "value": "human_baseline"},
                    {"label": "Alt 1 performer — interdependent", "value": "human_full_support"},
                    {"label": "Alt 2 performer — independent", "value": "agent_whenever_possible"},
                    {"label": "Alt 2 performer — interdependent", "value": "agent_whenever_possible_full_support"},
                    {"label": "Path of highest reliability", "value": "most_reliable"},
                ],
                value="none",
                labelStyle={"display": "inline-block", "marginRight": "20px"},
                style={"marginTop": "10px"},
            ),

            # Category Overrides
            html.Div(id="category-overrides-section", style={"display": "none"}, children=[
                html.H3("Category Overrides", style={"marginTop": "30px", "textTransform": "uppercase",
                                                      "letterSpacing": "0.04em"}),
                html.P(
                    "Click a bar to force a specific agent type for that category. "
                    "Click again to reset to default strategy.",
                    style={"fontSize": "14px"},
                ),
                html.Div(id="category-override-warning", style={"fontSize": "13px", "color": ACCENT, "marginTop": "6px"}),
                html.Div(id="category-overrides-container"),
            ]),

            # Automation Proportion Summary
            html.Div(id="automation-proportion-box", children=[
                html.Div([
                    html.Span("Automation Proportion: ",
                               style={"fontSize": "16px", "fontWeight": "bold"}),
                    html.Span(id="ap-summary-value", children="--",
                               style={"fontSize": "20px", "fontWeight": "bold", "color": ACCENT}),
                ], className="ap-box"),
                html.Div(id="ap-detail", style={"fontSize": "13px", "marginTop": "6px"}),
            ], style={"marginTop": "20px", "display": "none"}),
            html.Div(id="automation-proportion-results"),
            dcc.Graph(id="interdependence-graph", config={"displayModeBar": False}),

            # Team alternative labels (dynamic)
            html.Div(id="alt-labels"),

            # ── Choice Metrics ────────────────────────────────────────────
            html.Div(id="choice-metrics", style={"marginTop": "20px"}),

            # Workflow graph export buttons
            html.Div([
                html.Button("Export SVG", id="export-graph-svg-button", n_clicks=0),
                html.Button("Export PNG", id="export-graph-png-button", n_clicks=0),
                dcc.Download(id="download-graph-svg"),
                dcc.Download(id="download-graph-png"),
                html.Span(id="graph-export-status",
                          style={"marginLeft": "12px", "fontStyle": "italic", "fontSize": "13px"}),
            ], style={"display": "flex", "gap": "10px", "alignItems": "center", "marginTop": "12px"}),
        ]),

        # ── [04] Statistics ──
        html.Div(id="statistics-anchor", className="section", children=[
            html.Div("[ 04 ]", className="section-number"),
            html.H2("Statistics", className="section-title"),
            html.Div([
                html.Div([
                    dcc.Graph(id="allocation-type-bar-chart", config={"displayModeBar": False}),
                    html.Div([
                        html.Button("Export SVG", id="export-alloc-svg-button", n_clicks=0),
                        html.Button("Export PNG", id="export-alloc-png-button", n_clicks=0),
                        dcc.Download(id="download-alloc-svg"),
                        dcc.Download(id="download-alloc-png"),
                        html.Span(id="export-alloc-status",
                                  style={"marginLeft": "8px", "fontStyle": "italic", "fontSize": "13px"}),
                    ], style={"display": "flex", "gap": "10px", "alignItems": "center", "marginTop": "6px"}),
                ]),
                html.Div([
                    dcc.Graph(id="agent-autonomy-bar-chart", config={"displayModeBar": False}),
                    html.Div([
                        html.Button("Export SVG", id="export-autonomy-svg-button", n_clicks=0),
                        html.Button("Export PNG", id="export-autonomy-png-button", n_clicks=0),
                        dcc.Download(id="download-autonomy-svg"),
                        dcc.Download(id="download-autonomy-png"),
                        html.Span(id="export-autonomy-status",
                                  style={"marginLeft": "8px", "fontStyle": "italic", "fontSize": "13px"}),
                    ], style={"display": "flex", "gap": "10px", "alignItems": "center", "marginTop": "6px"}),
                ]),
            ]),
        ]),
    ]),

    # Footer
    html.Footer(
        "© Benjamin R. Berton 2025 Polytechnique Montreal",
        className="site-footer",
    ),
], style={
    "fontFamily": "var(--font-body, 'Space Grotesk', 'Inter', sans-serif)",
    "backgroundColor": BG,
    "color": INK,
    "margin": "0",
    "padding": "0",
})


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Preset buttons ────────────────────────────────────────────────────────────
@app.callback(
    Output("manual-columns-input", "value"),
    Output("manual-task-col-input", "value"),
    Output("manual-procedure-col-input", "value"),
    Input("preset-2agent", "n_clicks"),
    Input("preset-3agent", "n_clicks"),
    prevent_initial_call=True,
)
def apply_preset(n2, n3):
    ctx = callback_context
    btn = ctx.triggered[0]["prop_id"].split(".")[0]
    if btn == "preset-2agent":
        return "Human*, Robot, Robot*, Human", "Task", "Procedure"
    elif btn == "preset-3agent":
        return "Human*, UGV, UAV, UGV*, UAV*, Human", "Task", "Procedure"
    return dash.no_update, dash.no_update, dash.no_update


# ── Nav-links visibility ─────────────────────────────────────────────────────
@app.callback(
    Output("nav-links", "style"),
    Input("team-config-store", "data"),
)
def toggle_nav_links(config):
    return SHOW if config else HIDE


# ── Setup / Config callback ──────────────────────────────────────────────────
@app.callback(
    Output("team-config-store", "data"),
    Output("table-wrapper", "children"),
    Output("analysis-section", "style"),
    Output("setup-section", "style"),
    Output("config-summary", "style"),
    Output("config-details", "children"),
    Output("procedure-dropdown", "options"),
    Output("upload-status", "children"),
    # Hierarchy-panel outputs
    Output("pending-csv-store", "data"),
    Output("hierarchy-config-section", "style"),
    Output("hier-level-1", "options"),
    Output("hier-level-1", "value"),
    Output("hier-level-2", "options"),
    Output("hier-level-2", "value"),
    Output("hier-level-3", "options"),
    Output("hier-level-3", "value"),
    Output("hier-level-4", "options"),
    Output("hier-level-4", "value"),
    Input("setup-upload", "contents"),
    Input("create-team-button", "n_clicks"),
    Input("reset-config-button", "n_clicks"),
    Input("load-example-button", "n_clicks"),
    Input("confirm-hierarchy-button", "n_clicks"),
    State("setup-upload", "filename"),
    State("manual-columns-input", "value"),
    State("manual-task-col-input", "value"),
    State("manual-procedure-col-input", "value"),
    State("manual-category-col-input", "value"),
    State("pending-csv-store", "data"),
    State("hier-level-1", "value"),
    State("hier-level-2", "value"),
    State("hier-level-3", "value"),
    State("hier-level-4", "value"),
    prevent_initial_call=True,
)
def handle_setup(upload_contents, create_clicks, reset_clicks, example_clicks, confirm_clicks,
                 upload_filename, manual_columns, manual_task_col,
                 manual_procedure_col, manual_category_col,
                 pending_csv, hier_l1, hier_l2, hier_l3, hier_l4):
    ctx = callback_context
    triggered = ctx.triggered[0]["prop_id"].split(".")[0]
    no = dash.no_update
    # 18 outputs: 8 core + pending-csv-store + hierarchy-section + 4*(options+value)
    _no18 = (no,) * 18

    def _hide_hier():
        """Return the 10 hierarchy-panel outputs that hide/clear the panel."""
        return None, HIDE, [], None, [], None, [], None, [], None

    if triggered == "reset-config-button":
        return (None, None, HIDE, SHOW, HIDE, None, [], "") + _hide_hier()

    if triggered == "load-example-button":
        if EXAMPLE_CSV is None:
            return (no, no, no, no, no, no, no, "⚠️ Example file not found.") + _hide_hier()
        try:
            df = pd.read_csv(EXAMPLE_CSV)
        except Exception as e:
            return (no, no, no, no, no, no, no, f"⚠️ Error reading example: {e}") + _hide_hier()
        config = detect_team_config(df)
        if config is None:
            return (no, no, no, no, no, no, no, "⚠️ Could not detect team structure in example.") + _hide_hier()
        if "Row" not in df.columns:
            df.insert(0, "Row", range(1, len(df) + 1))
            config["all_columns"] = ["Row"] + [c for c in config["all_columns"] if c != "Row"]
        proc_col = config.get("procedure_column", "Procedure")
        proc_options = [{"label": p, "value": p} for p in df[proc_col].dropna().unique()] if proc_col in df.columns else []
        table = build_data_table(df, config)
        summary = config_summary_html(config)
        return (config, table, SHOW, HIDE, SHOW, summary, proc_options, "✅ Loaded example: IA_V8.csv") + _hide_hier()

    if triggered == "setup-upload" and upload_contents:
        # Step 1: parse CSV, auto-detect hierarchy, show the panel for user to confirm/adjust
        try:
            content_type, content_string = upload_contents.split(",")
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        except Exception as e:
            return (no, no, no, no, no, no, no, f"⚠️ Error reading file: {e}") + _hide_hier()

        config = detect_team_config(df)
        if config is None:
            return (no, no, no, no, no, no, no, "⚠️ Could not detect team structure. No color columns found.") + _hide_hier()

        # Build dropdown options from all non-color, non-Row columns
        color_set = set(config.get("color_columns", []))
        candidate_cols = [c for c in df.columns if c != "Row" and c not in color_set]
        options = [{"label": c, "value": c} for c in candidate_cols]

        hier = config.get("hierarchy_columns", [])
        v1 = hier[0] if len(hier) > 0 else None
        v2 = hier[1] if len(hier) > 1 else None
        v3 = hier[2] if len(hier) > 2 else None
        v4 = hier[3] if len(hier) > 3 else None

        n_hier = len(hier)
        pending = {"content": content_string, "filename": upload_filename or "file.csv"}
        status = (
            f"✅ {upload_filename} — detected {n_hier} hierarchy level(s): "
            f"{', '.join(hier)}. Adjust below if needed, then click Load."
        )
        return (
            no, no, no, no, no, no, no, status,
            pending, SHOW,
            options, v1, options, v2, options, v3, options, v4,
        )

    if triggered == "confirm-hierarchy-button":
        # Step 2: decode pending CSV, override hierarchy, build table
        if not pending_csv:
            return _no18
        try:
            decoded = base64.b64decode(pending_csv["content"])
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        except Exception as e:
            return (no, no, no, no, no, no, no, f"⚠️ Error reading pending file: {e}") + (no,) * 10

        config = detect_team_config(df)
        if config is None:
            return (no, no, no, no, no, no, no, "⚠️ Could not detect team structure.") + (no,) * 10

        # Override hierarchy with user-selected levels
        selected_hier = [l for l in [hier_l1, hier_l2, hier_l3, hier_l4] if l]
        if selected_hier:
            config["hierarchy_columns"] = selected_hier
            config["procedure_column"] = selected_hier[0]
            config["task_column"] = selected_hier[-1]
            # Recompute metadata (exclude selected hierarchy from metadata)
            color_set = set(config["color_columns"])
            structural = {"Row"} | set(selected_hier)
            if config.get("category_column"):
                structural.add(config["category_column"])
            config["metadata_columns"] = [
                c for c in df.columns if c not in structural and c not in color_set
            ]

        # Ensure Row column
        if "Row" not in df.columns:
            df.insert(0, "Row", range(1, len(df) + 1))
            config["all_columns"] = ["Row"] + [c for c in config["all_columns"] if c != "Row"]

        proc_col = config.get("procedure_column", "Procedure")
        proc_options = [{"label": p, "value": p} for p in df[proc_col].dropna().unique()] if proc_col in df.columns else []
        table = build_data_table(df, config)
        summary = config_summary_html(config)
        filename = pending_csv.get("filename", "file.csv")
        return (
            config, table, SHOW, HIDE, SHOW, summary, proc_options, f"✅ Loaded {filename}",
        ) + _hide_hier()

    if triggered == "create-team-button":
        config = build_config_from_manual(
            manual_columns or "",
            task_col=manual_task_col or "Task",
            procedure_col=manual_procedure_col or "Procedure",
            category_col=manual_category_col.strip() or None if manual_category_col else None,
        )
        if config is None:
            return _no18
        df = create_empty_df(config)
        table = build_data_table(df, config)
        summary = config_summary_html(config)
        return (config, table, SHOW, HIDE, SHOW, summary, [], "") + _hide_hier()

    return _no18




# ── Table operations callback ────────────────────────────────────────────────
@app.callback(
    Output("table-wrapper", "children", allow_duplicate=True),
    Output("save-confirmation", "children"),
    Output("download-csv", "data"),
    Output("procedure-dropdown", "options", allow_duplicate=True),
    Output("team-config-store", "data", allow_duplicate=True),
    Output("analysis-section", "style", allow_duplicate=True),
    Output("setup-section", "style", allow_duplicate=True),
    Output("config-summary", "style", allow_duplicate=True),
    Output("config-details", "children", allow_duplicate=True),
    Input("responsibility-table", "data"),
    Input("save-button", "n_clicks"),
    Input("upload-data", "contents"),
    Input("add-row-button", "n_clicks"),
    Input("copy-down-button", "n_clicks"),
    State("responsibility-table", "active_cell"),
    State("upload-data", "filename"),
    State("team-config-store", "data"),
    prevent_initial_call=True,
)
def handle_table(data, save_clicks, upload_contents, add_clicks, copy_clicks,
                 active_cell, upload_filename, config):
    ctx = callback_context
    triggered = ctx.triggered[0]["prop_id"].split(".")[0]
    no = dash.no_update

    save_msg = ""
    download = None

    # ── Save ──
    if triggered == "save-button" and data and config:
        df = pd.DataFrame(data)
        download = dcc.send_data_frame(df.to_csv, "interdependence_analysis.csv", index=False)
        save_msg = "✅ Table downloaded as interdependence_analysis.csv"
        return no, save_msg, download, no, no, no, no, no, no

    # ── Load new CSV (from analysis section) ──
    if triggered == "upload-data" and upload_contents:
        try:
            ct, cs = upload_contents.split(",")
            decoded = base64.b64decode(cs)
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        except Exception as e:
            return no, f"⚠️ Error: {e}", None, no, no, no, no, no, no

        new_config = detect_team_config(df)
        if new_config is None:
            return no, "⚠️ Could not detect team structure.", None, no, no, no, no, no, no

        if "Row" not in df.columns:
            df.insert(0, "Row", range(1, len(df) + 1))
            new_config["all_columns"] = ["Row"] + [c for c in new_config["all_columns"] if c != "Row"]

        proc_col = new_config.get("procedure_column", "Procedure")
        proc_options = []
        if proc_col in df.columns:
            proc_options = [{"label": p, "value": p} for p in df[proc_col].dropna().unique()]

        table = build_data_table(df, new_config)
        summary = config_summary_html(new_config)
        return table, f"✅ Loaded {upload_filename}", None, proc_options, new_config, SHOW, HIDE, SHOW, summary

    # ── Add Row ──
    if triggered == "add-row-button" and data and config:
        df = pd.DataFrame(data)
        new_row = {col: "" for col in df.columns}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df = ensure_row_column(df)
        table = build_data_table(df, config)
        return table, "", None, no, no, no, no, no, no

    # ── Copy Down ──
    if triggered == "copy-down-button" and data and config:
        df = pd.DataFrame(data)
        if active_cell and "row" in active_cell and "column_id" in active_cell:
            r = active_cell["row"]
            c = active_cell["column_id"]
            if r is not None and c is not None and r + 1 < len(df):
                df.at[r + 1, c] = df.at[r, c]
        table = build_data_table(df, config)
        return table, "", None, no, no, no, no, no, no

    # ── Table edited ──
    if triggered == "responsibility-table" and data and config:
        df = pd.DataFrame(data)
        df = ensure_row_column(df)
        proc_col = config.get("procedure_column", "Procedure")
        proc_options = []
        if proc_col in df.columns:
            proc_options = [{"label": p, "value": p} for p in df[proc_col].dropna().unique()]
        table = build_data_table(df, config)
        return table, "", None, proc_options, no, no, no, no, no

    return no, "", None, no, no, no, no, no, no


# ── Graph callbacks (two-stage: base figure + highlighting) ───────────────────
@app.callback(
    Output("base-figure-store", "data"),
    Output("arrow-indices-store", "data"),
    Output("alt-labels", "children"),
    Input("procedure-dropdown", "value"),
    Input("view-selector", "value"),
    Input("responsibility-table", "data"),
    State("team-config-store", "data"),
)
def build_base_figure(procedure, view_mode, data, config):
    """Build the base figure structure (without highlighting).
    Only runs when procedure, view mode, or data changes."""
    if not data or not config:
        return None, None, None

    df = pd.DataFrame(data)
    if df.empty:
        return None, None, None

    fig, arrow_info = build_workflow_figure_base(
        df, config, procedure=procedure, view_mode=view_mode or "full",
    )

    # Team alternative labels
    labels = []
    for alt in config.get("alternatives", []):
        labels.append(html.Div(
            alt["name"],
            style={
                "display": "inline-block", "textAlign": "center",
                "marginTop": "10px", "fontWeight": "bold",
                "flex": "1",
            },
        ))
    alt_label_div = html.Div(labels, style={
        "display": "flex", "width": "100%",
        "marginLeft": "150px", "marginRight": "50px",
    }) if labels else None

    return fig.to_dict(), arrow_info, alt_label_div


@app.callback(
    Output("interdependence-graph", "figure"),
    Input("base-figure-store", "data"),
    Input("arrow-indices-store", "data"),
    Input("highlight-selector", "value"),
    Input("category-overrides-store", "data"),
    Input("procedure-dropdown", "value"),
    State("responsibility-table", "data"),
    State("team-config-store", "data"),
)
def apply_highlighting_callback(base_fig_dict, arrow_info, highlight_track,
                                 category_overrides, procedure, data, config):
    """Apply highlighting to the cached base figure. Fast: only updates colors/widths."""
    if not base_fig_dict or not data or not config:
        return go.Figure()

    df = pd.DataFrame(data)
    ht = None if highlight_track == "none" else highlight_track

    return apply_workflow_highlighting(
        base_fig_dict, arrow_info, df, config,
        procedure=procedure, highlight_track=ht,
        category_overrides=category_overrides or {},
    )


# ── Bar chart callback ────────────────────────────────────────────────────────
@app.callback(
    Output("allocation-type-bar-chart", "figure"),
    Output("agent-autonomy-bar-chart", "figure"),
    Input("procedure-dropdown", "value"),
    Input("responsibility-table", "data"),
    State("team-config-store", "data"),
)
def update_bar_charts(procedure, data, config):
    if not data or not config:
        return go.Figure(), go.Figure()

    df = pd.DataFrame(data)
    proc_col = config.get("procedure_column", "Procedure")
    if procedure and proc_col in df.columns:
        df = df[df[proc_col] == procedure]

    return (
        build_allocation_bar_chart(df, config),
        build_autonomy_bar_chart(df, config),
    )


# ── Automation Proportion callback ────────────────────────────────────────────
@app.callback(
    Output("automation-proportion-box", "style"),
    Output("ap-summary-value", "children"),
    Output("ap-summary-value", "style"),
    Output("ap-detail", "children"),
    Output("automation-proportion-results", "children"),
    Input("highlight-selector", "value"),
    Input("category-overrides-store", "data"),
    Input("procedure-dropdown", "value"),
    State("responsibility-table", "data"),
    State("team-config-store", "data"),
)
def compute_automation_proportion(highlight_track, category_overrides, procedure, data, config):
    if not data or not config:
        return {"display": "none"}, "--", {}, "", None

    df = pd.DataFrame(data)
    proc_col = config.get("procedure_column", "Procedure")
    if procedure and proc_col in df.columns:
        df = df[df[proc_col] == procedure]

    cat_col = config.get("category_column") or "Category"
    if cat_col not in df.columns or not highlight_track or highlight_track == "none":
        return {"display": "none"}, "--", {}, "", None

    P_min, P_max, cat_scores = compute_automation_proportion_data(
        df, config, highlight_track, category_overrides or {},
    )

    if P_min is None:
        return {"display": "none"}, "--", {}, "", None

    is_range = abs(P_max - P_min) > 1e-6

    # Color-code using midpoint
    P_mid = (P_min + P_max) / 2
    if P_mid > 0.55:
        color = ACCENT
    elif P_mid < 0.45:
        color = PAL_GREEN
    else:
        color = PAL_YELLOW

    val_style = {"fontSize": "20px", "fontWeight": "bold", "color": color}
    box_style = {"textAlign": "center", "marginTop": "10px"}

    # Summary value: range or single
    if is_range:
        summary_text = f"{P_min:.3f} – {P_max:.3f}"
    else:
        summary_text = f"{P_min:.3f}"

    # Category detail with ranges
    detail_parts = []
    for cat, (pk_min, pk_max) in sorted(cat_scores.items()):
        if abs(pk_max - pk_min) > 1e-6:
            detail_parts.append(f"{cat}: {pk_min:.2f}–{pk_max:.2f}")
        else:
            detail_parts.append(f"{cat}: {pk_min:.2f}")
    detail_text = " | ".join(detail_parts) if detail_parts else ""

    # Extract performer names for formula explanation
    alts = config.get("alternatives", [])
    def _perf_name(alt):
        perfs = alt.get("performers", [])
        return perfs[0].rstrip("*") if perfs else "Agent"
    alt1_name = _perf_name(alts[0]) if len(alts) > 0 else "Alt 1 performer"
    alt2_name = _perf_name(alts[1]) if len(alts) > 1 else "Alt 2 performer"

    sum_min = sum(v[0] for v in cat_scores.values())
    sum_max = sum(v[1] for v in cat_scores.values())
    K = len(cat_scores)

    if is_range:
        formula_p = (
            f"P = (1/K) × Σ pₖ = (1/{K}) × [{sum_min:.2f}, {sum_max:.2f}] "
            f"= [{P_min:.3f}, {P_max:.3f}]"
        )
    else:
        formula_p = (
            f"P = (1/K) × Σ pₖ = (1/{K}) × {sum_min:.2f} = {P_min:.3f}"
        )

    formula = html.Div([
        html.P([
            html.B("Formula: "), formula_p,
        ], style={"fontSize": "13px", "marginTop": "5px"}),
        html.P([
            html.B("Where: "),
            f"w(t) = 0.0 ({alt1_name} alone), "
            f"[0.0, 0.5] ({alt1_name} with optional auto-support), "
            f"0.5 ({alt1_name} with mandatory support), "
            f"[0.75, 1.0] ({alt2_name} with optional human-support), "
            f"0.75 ({alt2_name} with mandatory human-support), "
            f"1.0 ({alt2_name} alone)",
        ], style={"fontSize": "12px"}),
        html.P(
            "Range shown when support is opportunistic (green/yellow); "
            "orange performers/supporters indicate mandatory support (fixed value).",
            style={"fontSize": "11px", "color": INK_MUTED, "marginTop": "4px"},
        ),
    ])

    return box_style, summary_text, val_style, detail_text, formula


# ── Dynamic highlight-selector labels ─────────────────────────────────────────
@app.callback(
    Output("highlight-selector", "options"),
    Input("team-config-store", "data"),
)
def update_highlight_options(config):
    def _perf_name(alt):
        perfs = alt.get("performers", [])
        return perfs[0].rstrip("*") if perfs else "Agent"
    if not config or not config.get("alternatives"):
        a1, a2 = "Alt 1 performer", "Alt 2 performer"
    else:
        alts = config["alternatives"]
        a1 = _perf_name(alts[0]) if len(alts) > 0 else "Alt 1 performer"
        a2 = _perf_name(alts[1]) if len(alts) > 1 else "Alt 2 performer"
    return [
        {"label": "No highlight", "value": "none"},
        {"label": f"{a1} — independent", "value": "human_baseline"},
        {"label": f"{a1} — interdependent", "value": "human_full_support"},
        {"label": f"{a2} — independent", "value": "agent_whenever_possible"},
        {"label": f"{a2} — interdependent", "value": "agent_whenever_possible_full_support"},
        {"label": "Path of highest reliability", "value": "most_reliable"},
    ]


# ── Category Overrides callbacks ──────────────────────────────────────────────
@app.callback(
    Output("category-overrides-section", "style"),
    Output("category-overrides-container", "children"),
    Input("responsibility-table", "data"),
    State("team-config-store", "data"),
)
def generate_category_overrides(data, config):
    """Generate per-category override UI with mini bar charts."""
    if not data or not config:
        return {"display": "none"}, None

    df = pd.DataFrame(data)
    cat_col = config.get("category_column") or "Category"
    if cat_col not in df.columns:
        return {"display": "none"}, None

    agent_types = {a["name"]: a["type"] for a in config["agents"]}
    performer_cols = get_performer_columns(config)
    categories = sorted(df[cat_col].dropna().unique())

    if not categories:
        return {"display": "none"}, None

    # Group performers by type
    human_perfs = [
        pc for pc in performer_cols
        if agent_types.get(pc.rstrip("*"), "").lower() == "human"
    ]
    auto_perfs = [
        pc for pc in performer_cols
        if agent_types.get(pc.rstrip("*"), "").lower() == "autonomous"
    ]

    if not human_perfs or not auto_perfs:
        return {"display": "none"}, None

    human_label = "Human"
    auto_label = ", ".join([pc.rstrip("*") for pc in auto_perfs])

    children = []
    for cat in categories:
        cat_df = df[df[cat_col] == cat]

        human_assignable = sum(
            1 for _, row in cat_df.iterrows()
            for pc in human_perfs
            if pc in df.columns
            and str(row.get(pc, "") or "").strip().lower() in ("green", "yellow", "orange")
        )
        auto_assignable = sum(
            1 for _, row in cat_df.iterrows()
            for pc in auto_perfs
            if pc in df.columns
            and str(row.get(pc, "") or "").strip().lower() in ("green", "yellow", "orange")
        )

        fig = go.Figure()
        max_count = max(auto_assignable, human_assignable, 1)
        _ts = time.time()
        fig.add_trace(go.Bar(
            y=[auto_label], x=[auto_assignable], orientation="h",
            marker_color="#555250", name=auto_label,
            text=[f"{auto_assignable}"], textposition="outside",
            cliponaxis=False, customdata=[_ts],
        ))
        fig.add_trace(go.Bar(
            y=[human_label], x=[human_assignable], orientation="h",
            marker_color="#555250", name=human_label,
            text=[f"{human_assignable}"], textposition="outside",
            cliponaxis=False, customdata=[_ts],
        ))
        fig.update_layout(
            title=f"{cat} ({len(cat_df)} tasks)",
            height=100, margin=dict(l=80, r=40, t=25, b=5),
            showlegend=False, barmode="group",
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                       range=[0, max_count * 1.5]),
            yaxis=dict(showgrid=False),
            plot_bgcolor=BG, paper_bgcolor=BG,
            font=dict(family="Space Grotesk, Inter, sans-serif", color=INK),
        )

        children.append(html.Div([
            dcc.Graph(
                id={"type": "category-bar-chart", "category": cat},
                figure=fig, config={"displayModeBar": False},
                style={"height": "100px"},
            ),
            dcc.Store(id={"type": "category-override", "category": cat}, data="default"),
            dcc.Store(id={"type": "category-counts", "category": cat},
                      data={"human": human_assignable, "auto": auto_assignable,
                            "auto_label": auto_label,
                            "title": f"{cat} ({len(cat_df)} tasks)"}),
        ], style={
            "display": "inline-block", "width": "220px",
            "verticalAlign": "top", "margin": "4px",
            "border": f"1px solid {BORDER}", "padding": "3px",
            "backgroundColor": SURFACE,
        }))

    return {"display": "block"}, children


@app.callback(
    Output({"type": "category-override", "category": dash.MATCH}, "data"),
    Output({"type": "category-bar-chart", "category": dash.MATCH}, "figure"),
    Input({"type": "category-bar-chart", "category": dash.MATCH}, "clickData"),
    State({"type": "category-override", "category": dash.MATCH}, "data"),
    State({"type": "category-counts", "category": dash.MATCH}, "data"),
    State("highlight-selector", "value"),
    prevent_initial_call=True,
)
def toggle_category_selection(click_data, current_selection, counts, highlight_value):
    """Toggle category override when clicking a bar."""
    if not click_data:
        return dash.no_update, dash.no_update

    if not highlight_value or highlight_value == "none":
        return dash.no_update, dash.no_update

    clicked_label = click_data["points"][0].get("y", None)
    if clicked_label is None:
        return dash.no_update, dash.no_update

    auto_label = counts.get("auto_label", "Autonomous")

    # Map clicked label to agent type
    if clicked_label == auto_label:
        clicked_type = "autonomous"
    elif clicked_label == "Human":
        clicked_type = "human"
    else:
        return dash.no_update, dash.no_update

    # Toggle: clicking same bar deselects
    new_selection = "default" if current_selection == clicked_type else clicked_type

    # Rebuild figure with updated colors
    human_color = ACCENT if new_selection == "human" else "#555250"
    auto_color = ACCENT if new_selection == "autonomous" else "#555250"

    auto_count = counts.get("auto", 0)
    human_count = counts.get("human", 0)
    max_count = max(auto_count, human_count, 1)
    _ts = time.time()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[auto_label], x=[auto_count], orientation="h",
        marker_color=auto_color, name=auto_label,
        text=[f"{auto_count}"], textposition="outside",
        cliponaxis=False, customdata=[_ts],
    ))
    fig.add_trace(go.Bar(
        y=["Human"], x=[human_count], orientation="h",
        marker_color=human_color, name="Human",
        text=[f"{human_count}"], textposition="outside",
        cliponaxis=False, customdata=[_ts],
    ))
    fig.update_layout(
        title=counts.get("title", ""),
        height=100, margin=dict(l=80, r=40, t=25, b=5),
        showlegend=False, barmode="group",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   range=[0, max_count * 1.5]),
        yaxis=dict(showgrid=False),
        plot_bgcolor=BG, paper_bgcolor=BG,
        font=dict(family="Space Grotesk, Inter, sans-serif", color=INK),
    )

    return new_selection, fig


@app.callback(
    Output("category-override-warning", "children"),
    Input("highlight-selector", "value"),
)
def show_category_override_warning(highlight_value):
    """Show a warning when no highlight track is selected."""
    if not highlight_value or highlight_value == "none":
        return (
            "Select a highlight strategy first. Automation proportion requires a track "
            "to compute against — pick a highlight above, or specify every category manually."
        )
    return ""


@app.callback(
    Output("category-overrides-store", "data"),
    Input({"type": "category-override", "category": dash.ALL}, "data"),
    State({"type": "category-override", "category": dash.ALL}, "id"),
    prevent_initial_call=True,
)
def collect_category_overrides(values, ids):
    """Collect all category overrides into a single store."""
    overrides = {}
    for id_dict, value in zip(ids, values):
        if value != "default":
            overrides[id_dict["category"]] = value
    return overrides


# ── Choice Metrics callback ───────────────────────────────────────────────────
@app.callback(
    Output("choice-metrics", "children"),
    Input("responsibility-table", "data"),
    Input("procedure-dropdown", "value"),
    State("team-config-store", "data"),
)
def compute_choice_metrics(data, procedure, config):
    """
    For each task: count the number of valid (non-red) performer columns.
    A task with exactly 1 performer has no choice to make (it's forced).
    A task with N >= 2 performers offers N choices.

    Individual choices = sum of performer counts for tasks where count >= 2.
    Combinations       = product of performer counts across all tasks (each
                         forced task contributes factor 1, so it's excluded
                         from the product automatically).
    """
    if not data or not config:
        return None

    df = pd.DataFrame(data)
    proc_col = config.get("procedure_column", "Procedure")
    if procedure and proc_col in df.columns:
        df = df[df[proc_col] == procedure]

    if df.empty:
        return None

    performer_cols = get_performer_columns(config)

    # performer_count[i] = number of valid (non-red) performers for task i
    performer_count = []
    for _, row in df.iterrows():
        n = sum(
            1 for pc in performer_cols
            if pc in df.columns
            and str(row.get(pc, "") or "").strip().lower() in ("green", "yellow", "orange")
        )
        performer_count.append(n)

    total_tasks = len(performer_count)

    # Only tasks with >= 2 options involve an actual choice
    total_individual = sum(n for n in performer_count if n >= 2)

    # Product: only tasks with >= 2 options expand the combination space
    product = 1
    for n in performer_count:
        if n >= 2:
            product *= n

    # Distribution buckets:
    #   0 performers  → unassigned (no performer available)
    #   1 performer   → forced (no choice)
    #   N >= 2        → N choices
    from collections import Counter
    dist = Counter(performer_count)
    dist_items = sorted(dist.items())  # [(n_performers, n_tasks), ...]

    # Format the combination count
    import math
    if product > 1e15:
        combo_display = f"{product:.3e}"
        combo_sub = f"(log₁₀ = {math.log10(product):.1f})"
    else:
        combo_display = f"{product:,}"
        combo_sub = ""

    # Distribution cells
    dist_cells = []
    for n_perf, n_tasks in dist_items:
        if n_perf == 0:
            label = "no performer"
            color = ACCENT
        elif n_perf == 1:
            label = "forced (1 performer)"
            color = INK_MUTED
        else:
            label = f"{n_perf} choices"
            color = PAL_ORANGE if n_perf == 2 else PAL_GREEN
        dist_cells.append(html.Div([
            html.Span(str(n_tasks),
                      style={"fontSize": "22px", "fontWeight": "bold", "color": color}),
            html.Br(),
            html.Span(f"task{'s' if n_tasks != 1 else ''} — {label}",
                      style={"fontSize": "11px", "color": INK_MUTED}),
        ], style={
            "textAlign": "center", "minWidth": "120px",
            "borderRight": f"1px solid {BORDER}", "padding": "0 18px",
        }))

    return html.Div([
        html.Div("Allocation Choices", style={
            "fontSize": "11px", "fontWeight": "bold", "letterSpacing": "0.06em",
            "textTransform": "uppercase", "color": INK_MUTED, "marginBottom": "10px",
        }),
        html.Div([
            # Big metrics
            html.Div([
                html.Span(str(total_individual),
                          style={"fontSize": "28px", "fontWeight": "bold", "color": INK}),
                html.Br(),
                html.Span("individual choices",
                          style={"fontSize": "11px", "color": INK_MUTED}),
            ], style={"textAlign": "center", "minWidth": "130px",
                      "borderRight": f"1px solid {BORDER}", "padding": "0 18px"}),

            html.Div([
                html.Span(combo_display,
                          style={"fontSize": "28px", "fontWeight": "bold", "color": INK}),
                html.Br(),
                html.Span("total combinations",
                          style={"fontSize": "11px", "color": INK_MUTED}),
                html.Br() if combo_sub else None,
                html.Span(combo_sub,
                          style={"fontSize": "10px", "color": INK_MUTED}) if combo_sub else None,
            ], style={"textAlign": "center", "minWidth": "140px",
                      "borderRight": f"1px solid {BORDER}", "padding": "0 18px"}),

            html.Div([
                html.Span(str(total_tasks),
                          style={"fontSize": "28px", "fontWeight": "bold", "color": INK}),
                html.Br(),
                html.Span("tasks",
                          style={"fontSize": "11px", "color": INK_MUTED}),
            ], style={"textAlign": "center", "minWidth": "80px",
                      "borderRight": f"1px solid {BORDER}", "padding": "0 18px"}),

            # Per-choice-count distribution
            *dist_cells,
        ], style={
            "display": "flex", "alignItems": "center", "flexWrap": "wrap",
            "gap": "4px",
        }),
    ], style={
        "backgroundColor": SURFACE,
        "border": f"1px solid {BORDER}",
        "borderLeft": f"4px solid {INK}",
        "padding": "14px 20px",
        "borderRadius": "3px",
    })


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Table: Export PNG (server-side via kaleido) ───────────────────────────────
@app.callback(
    Output("download-table-png", "data"),
    Input("export-table-png-button", "n_clicks"),
    State("responsibility-table", "data"),
    State("responsibility-table", "hidden_columns"),
    State("team-config-store", "data"),
    prevent_initial_call=True,
)
def export_table_png(n_clicks, data, hidden_columns, config):
    if not n_clicks or not data or not config:
        return dash.no_update
    import plotly.io as pio

    df = pd.DataFrame(data)
    agent_cols = set(get_agent_columns(config))

    # Fixed export columns: all hierarchy levels + agent cols + teaming requirements
    hier_cols = config.get("hierarchy_columns", [
        config.get("procedure_column", "Procedure"),
        config.get("task_column", "Task Object"),
    ])
    TEAMING_COLS = ["Observability", "Predictability", "Directability"]

    all_cols = config.get("all_columns", [])
    export_cols = (
        [c for c in hier_cols if c in df.columns]
        + ([config.get("category_column")] if config.get("category_column") and config["category_column"] in df.columns else [])
        + [c for c in all_cols if c in agent_cols and c in df.columns]
        + [c for c in TEAMING_COLS if c in df.columns]
    )
    # Fallback if nothing matched
    if not export_cols:
        export_cols = [c for c in df.columns]

    visible_cols = export_cols
    df_vis = df[visible_cols]

    # Per-cell background colors (list-of-columns → list-of-rows)
    cell_bg = []
    cell_fg = []
    for col in visible_cols:
        bg_col, fg_col = [], []
        for _, row in df_vis.iterrows():
            if col in agent_cols:
                val = str(row.get(col, "") or "").strip().lower()
                bg = COLOR_MAP.get(val, BG) if val in VALID_COLORS else BG
                fg = BG if val in ("red", "orange", "green") else INK
            else:
                bg, fg = BG, INK
            bg_col.append(bg)
            fg_col.append(fg)
        cell_bg.append(bg_col)
        cell_fg.append(fg_col)

    fig = go.Figure(go.Table(
        header=dict(
            values=["<b>" + c + "</b>" for c in visible_cols],
            fill_color=INK,
            font=dict(color=BG, size=11, family="Arial, sans-serif"),
            align="center",
            height=30,
        ),
        cells=dict(
            values=[df_vis[col].tolist() for col in visible_cols],
            fill_color=cell_bg,
            font=dict(color=cell_fg, size=10, family="Arial, sans-serif"),
            align="left",
            height=25,
        ),
    ))
    n_rows = len(df_vis)
    n_cols = len(visible_cols)
    col_w = max(60, 1400 // max(n_cols, 1))
    width = min(max(800, n_cols * col_w), 2600)
    height = max(300, 50 + n_rows * 28)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor=BG,
                      width=width, height=height)
    img_bytes = pio.to_image(fig, format="png", width=width, height=height)
    return dcc.send_bytes(img_bytes, "interdependence_analysis.png")


# ── Table: Copy as Markdown (clientside) ─────────────────────────────────────
app.clientside_callback(
    """
    function(n_clicks, data, columns, hidden_cols) {
        if (!n_clicks || !data || !columns) return '';
        var hiddenSet = {};
        (hidden_cols || []).forEach(function(h) { hiddenSet[h] = true; });
        var visCols = columns.filter(function(c) { return !hiddenSet[c.id]; });
        var colNames = visCols.map(function(c) {
            var n = c.name;
            return Array.isArray(n) ? n[n.length - 1] : String(n);
        });
        var colIds = visCols.map(function(c) { return c.id; });
        var lines = [
            '| ' + colNames.join(' | ') + ' |',
            '| ' + colNames.map(function() { return '---'; }).join(' | ') + ' |'
        ];
        data.forEach(function(row) {
            var cells = colIds.map(function(id) {
                var v = row[id];
                return v == null ? '' : String(v).replace(/\\|/g, '\\\\|');
            });
            lines.push('| ' + cells.join(' | ') + ' |');
        });
        var md = lines.join('\\n');
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(md);
        } else {
            var ta = document.createElement('textarea');
            ta.value = md; ta.style.position = 'fixed'; ta.style.opacity = '0';
            document.body.appendChild(ta); ta.select();
            document.execCommand('copy'); document.body.removeChild(ta);
        }
        return 'Copied to clipboard!';
    }
    """,
    Output("copy-markdown-status", "children"),
    Input("copy-markdown-button", "n_clicks"),
    State("responsibility-table", "data"),
    State("responsibility-table", "columns"),
    State("responsibility-table", "hidden_columns"),
    prevent_initial_call=True,
)

# ── Workflow Graph: Export SVG (server-side via kaleido) ─────────────────────
@app.callback(
    Output("download-graph-svg", "data"),
    Output("graph-export-status", "children"),
    Input("export-graph-svg-button", "n_clicks"),
    State("interdependence-graph", "figure"),
    prevent_initial_call=True,
)
def export_graph_svg(n_clicks, figure):
    if not n_clicks or not figure:
        return dash.no_update, dash.no_update
    import plotly.io as pio
    fig = go.Figure(figure)
    svg_bytes = pio.to_image(fig, format="svg", width=1400, height=900)
    return dcc.send_bytes(svg_bytes, "workflow_graph.svg"), ""


# ── Workflow Graph: Export PNG (server-side via kaleido) ─────────────────────
@app.callback(
    Output("download-graph-png", "data"),
    Output("graph-export-status", "children", allow_duplicate=True),
    Input("export-graph-png-button", "n_clicks"),
    State("interdependence-graph", "figure"),
    prevent_initial_call=True,
)
def export_graph_png(n_clicks, figure):
    if not n_clicks or not figure:
        return dash.no_update, dash.no_update
    import plotly.io as pio
    fig = go.Figure(figure)
    png_bytes = pio.to_image(fig, format="png", width=1400, height=900, scale=2)
    return dcc.send_bytes(png_bytes, "workflow_graph.png"), ""


# ── Statistics: Allocation chart export (server-side via kaleido) ─────────────
@app.callback(
    Output("download-alloc-svg", "data"),
    Output("export-alloc-status", "children"),
    Input("export-alloc-svg-button", "n_clicks"),
    State("allocation-type-bar-chart", "figure"),
    prevent_initial_call=True,
)
def export_alloc_svg(n_clicks, figure):
    if not n_clicks or not figure:
        return dash.no_update, dash.no_update
    import plotly.io as pio
    fig = go.Figure(figure)
    svg_bytes = pio.to_image(fig, format="svg", width=1000, height=600)
    return dcc.send_bytes(svg_bytes, "task_type_distribution.svg"), ""


@app.callback(
    Output("download-alloc-png", "data"),
    Output("export-alloc-status", "children", allow_duplicate=True),
    Input("export-alloc-png-button", "n_clicks"),
    State("allocation-type-bar-chart", "figure"),
    prevent_initial_call=True,
)
def export_alloc_png(n_clicks, figure):
    if not n_clicks or not figure:
        return dash.no_update, dash.no_update
    import plotly.io as pio
    fig = go.Figure(figure)
    png_bytes = pio.to_image(fig, format="png", width=1000, height=600, scale=2)
    return dcc.send_bytes(png_bytes, "task_type_distribution.png"), ""


# ── Statistics: Autonomy chart export (server-side via kaleido) ───────────────
@app.callback(
    Output("download-autonomy-svg", "data"),
    Output("export-autonomy-status", "children"),
    Input("export-autonomy-svg-button", "n_clicks"),
    State("agent-autonomy-bar-chart", "figure"),
    prevent_initial_call=True,
)
def export_autonomy_svg(n_clicks, figure):
    if not n_clicks or not figure:
        return dash.no_update, dash.no_update
    import plotly.io as pio
    fig = go.Figure(figure)
    svg_bytes = pio.to_image(fig, format="svg", width=1000, height=600)
    return dcc.send_bytes(svg_bytes, "agent_autonomy.svg"), ""


@app.callback(
    Output("download-autonomy-png", "data"),
    Output("export-autonomy-status", "children", allow_duplicate=True),
    Input("export-autonomy-png-button", "n_clicks"),
    State("agent-autonomy-bar-chart", "figure"),
    prevent_initial_call=True,
)
def export_autonomy_png(n_clicks, figure):
    if not n_clicks or not figure:
        return dash.no_update, dash.no_update
    import plotly.io as pio
    fig = go.Figure(figure)
    png_bytes = pio.to_image(fig, format="png", width=1000, height=600, scale=2)
    return dcc.send_bytes(png_bytes, "agent_autonomy.png"), ""


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
