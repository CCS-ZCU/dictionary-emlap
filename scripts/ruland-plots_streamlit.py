# streamlit_app.py
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import fnmatch

import instances_embeddings_eval as iee



# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="Ruland in Usage", layout="wide")

INDEX_PARQUET = "../data/large_files/ruland-emlap-grela.parquet"
HITS_DIR = Path("../data/large_files/emlap_ruland_instances/")  # folder with per-entry parquet hit files


# ---------------------------
# Cached loaders
# ---------------------------
@st.cache_data(show_spinner=True)
def load_emlap_metadata():
    emlap_metadata = pd.read_csv(
        "https://raw.githubusercontent.com/CCS-ZCU/EMLAP_ETL/refs/heads/master/data/emlap_metadata.csv",
        sep=";",
    )

    # label used in plots
    emlap_metadata["labeldate"] = emlap_metadata.apply(
        lambda row: f"{row['working_title']} ({row['date_publication']})",
        axis=1,
    )

    # Build "Noscemus IDs that are represented in EMLAP"
    if "if_noscemus_id" in emlap_metadata.columns:
        noscemus_emlap_ids = [
            str(int(x)) for x in emlap_metadata["if_noscemus_id"].unique() if pd.notna(x)
        ]
    else:
        noscemus_emlap_ids = []

    return emlap_metadata, set(noscemus_emlap_ids)


@st.cache_data(show_spinner=True)
def load_lexeme_index():
    df = pd.read_parquet(INDEX_PARQUET)

    # keep only what we need for searching + counts + hit-file loading
    keep = [
        "Lemma",
        "instance_fname",
        "emlap_instances_N",
        "noscemus_instances_N",
        "cc_instances_N",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].drop_duplicates(subset=["Lemma"]).copy()

    df["Lemma"] = df["Lemma"].astype("string")
    if "instance_fname" in df.columns:
        df["instance_fname"] = df["instance_fname"].astype("string")

    # counts as ints (safe)
    for c in ["emlap_instances_N", "noscemus_instances_N", "cc_instances_N"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    return df


@st.cache_data(show_spinner=True)
def load_hits_df(fname: str) -> pd.DataFrame:
    """Load full hit file for an entry. Cached per fname."""
    if not fname:
        return pd.DataFrame()
    path = HITS_DIR / fname
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


# ---------------------------
# Helpers: ID parsing & filtering
# ---------------------------
def corpus_of_grela(grela_id: str) -> str:
    if not isinstance(grela_id, str):
        return "other"
    if grela_id.startswith("emlap"):
        return "emlap"
    if grela_id.startswith("noscemus"):
        return "noscemus"
    if grela_id.startswith("cc"):
        return "cc"
    return "other"


def _extract_after_prefix(grela_id: str, prefix: str) -> str | None:
    """
    Extract ID part after prefixes like:
      emlap_123, emlap:123, noscemus_456, noscemus:456
    """
    if not isinstance(grela_id, str):
        return None
    m = re.match(rf"^{re.escape(prefix)}[:_](.+)$", grela_id)
    return m.group(1) if m else None


def emlap_work_id(grela_id: str) -> str | None:
    return _extract_after_prefix(grela_id, "emlap")


def noscemus_id(grela_id: str) -> str | None:
    raw = _extract_after_prefix(grela_id, "noscemus")
    if raw is None:
        return None
    # usually numeric; normalize like your list does (string of int)
    m = re.match(r"^(\d+)", raw)
    return str(int(m.group(1))) if m else raw


def filter_hits_for_table(
    hits_df: pd.DataFrame,
    show_emlap: bool,
    show_nosc: bool,
    show_cc: bool,
    noscemus_emlap_ids: set[str],
    drop_emlap_overlap_from_noscemus: bool = True,
) -> pd.DataFrame:
    if hits_df.empty or "grela_id" not in hits_df.columns:
        return pd.DataFrame()

    df = hits_df.copy()
    df["corpus"] = df["grela_id"].astype("string").map(corpus_of_grela)

    # optional: remove noscemus hits that overlap with emlap (via emlap_metadata.if_noscemus_id)
    if drop_emlap_overlap_from_noscemus:
        nos_ids = df.loc[df["corpus"] == "noscemus", "grela_id"].astype("string").map(noscemus_id)
        overlap_mask = (df["corpus"] == "noscemus") & nos_ids.isin(noscemus_emlap_ids)
        df = df.loc[~overlap_mask].copy()

    mask = (
        (show_emlap & (df["corpus"] == "emlap")) |
        (show_nosc & (df["corpus"] == "noscemus")) |
        (show_cc   & (df["corpus"] == "cc"))
    )
    return df.loc[mask].copy()


def find_matching_lemmas(pattern: str, lexeme_index: pd.DataFrame):
    """
    Return list of dicts with Lemma + counts + instance_fname, sorted by emlap_instances_N desc.
    Supports '*' wildcard (fnmatch).
    """
    df = lexeme_index.dropna(subset=["Lemma"]).copy()
    df["Lemma"] = df["Lemma"].astype(str)

    if pattern is None:
        pattern = ""

    pattern = pattern.strip()
    if pattern == "":
        return []

    if "*" in pattern:
        rex = fnmatch.translate(pattern)
        df = df[df["Lemma"].str.contains(rex, case=False, regex=True, na=False)]
    else:
        df = df[df["Lemma"].str.contains(pattern, case=False, regex=False, na=False)]

    sort_col = "emlap_instances_N" if "emlap_instances_N" in df.columns else "Lemma"
    df = df.sort_values(sort_col, ascending=False)

    cols = ["Lemma", "instance_fname", "emlap_instances_N", "noscemus_instances_N", "cc_instances_N"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].to_dict("records")


# ---------------------------
# Plotting from hits
# ---------------------------
def compute_emlap_distributions_from_hits(hits_df: pd.DataFrame, emlap_metadata: pd.DataFrame):
    """
    Build:
      - bidecade grouped df: Bidecade, Frequency, tokens_N, RelativeFrequency
      - per-work df: metadata + Frequency + RelativeFrequency
    from *hits_df* (EMLAP subset only).
    """
    if hits_df.empty or "grela_id" not in hits_df.columns:
        return None, None

    em = hits_df[hits_df["grela_id"].astype("string").str.startswith("emlap", na=False)].copy()
    if em.empty:
        return None, None

    em["work_id"] = em["grela_id"].astype("string").map(emlap_work_id).astype("string")
    counts = Counter(em["work_id"].dropna().astype(str).tolist())

    md = emlap_metadata.copy()
    md["Frequency"] = md["no."].astype(str).map(counts).fillna(0).astype(int)
    md["RelativeFrequency"] = md["Frequency"] / md["tokens_N"]

    def bidecade(year):
        if pd.isna(year):
            return None
        y = int(year)
        start = (y // 20) * 20
        return f"{start}-{start+19}"

    md["Bidecade"] = md["date_publication"].apply(bidecade)

    grouped = (
        md.groupby("Bidecade", dropna=True)[["Frequency", "tokens_N"]]
          .sum()
          .reset_index()
          .sort_values(by="Bidecade", key=lambda c: c.str.extract(r"(\d+)")[0].astype(int))
    )
    grouped["RelativeFrequency"] = grouped["Frequency"] / grouped["tokens_N"]

    return grouped, md


def plot_emlap_fourpanel(entry: str, grouped, perwork):
    fig, axes = plt.subplots(2, 2, figsize=(30, 12), dpi=250)

    # Abs by bidecade
    ax1 = axes[0, 0]
    ax1.bar(grouped["Bidecade"], grouped["Frequency"], color="darkblue")
    ax1.set_xlabel("Bidecades", fontsize=11)
    ax1.set_ylabel("Absolute Frequency", fontsize=11)
    ax1.set_title(f"Absolute Frequency of '{entry}' by Bidecades", fontsize=13)
    ax1.set_xticks(range(len(grouped["Bidecade"])))
    ax1.set_xticklabels(grouped["Bidecade"], rotation=90, fontsize=9)

    # Rel by bidecade
    ax2 = axes[0, 1]
    ax2.bar(grouped["Bidecade"], grouped["RelativeFrequency"], color="darkgreen")
    ax2.set_xlabel("Bidecades", fontsize=11)
    ax2.set_ylabel("Relative Frequency (Frequency / tokens_N)", fontsize=11)
    ax2.set_title(f"Relative Frequency of '{entry}' by Bidecades", fontsize=13)
    ax2.set_xticks(range(len(grouped["Bidecade"])))
    ax2.set_xticklabels(grouped["Bidecade"], rotation=90, fontsize=9)

    # Abs by works
    ax3 = axes[1, 0]
    ax3.bar(perwork["labeldate"], perwork["Frequency"], color="darkblue")
    ax3.set_xlabel("Works", fontsize=11)
    ax3.set_ylabel("Absolute Frequency", fontsize=11)
    ax3.set_title(f"Absolute Frequency of '{entry}' by Works", fontsize=13)
    ax3.set_xticks(range(len(perwork["labeldate"])))
    ax3.set_xticklabels(perwork["labeldate"], rotation=90, fontsize=8)

    # Rel by works
    ax4 = axes[1, 1]
    ax4.bar(perwork["labeldate"], perwork["RelativeFrequency"], color="darkgreen")
    ax4.set_xlabel("Works", fontsize=11)
    ax4.set_ylabel("Relative Frequency (Frequency / tokens_N)", fontsize=11)
    ax4.set_title(f"Relative Frequency of '{entry}' by Works", fontsize=13)
    ax4.set_xticks(range(len(perwork["labeldate"])))
    ax4.set_xticklabels(perwork["labeldate"], rotation=90, fontsize=8)

    plt.tight_layout()
    return fig


# ---------------------------
# UI
# ---------------------------
st.title("Ruland in Usage")
st.markdown(
    """
This app visualizes the distribution of instances from Ruland's entries across **EMLAP works**
(absolute + relative), and lets you browse full cross-corpus hits (**EMLAP / Noscemus / Corpus Corporum**)
when available.

Source code: https://github.com/CCS-ZCU/dictionary-emlap
"""
)

emlap_metadata, noscemus_emlap_ids = load_emlap_metadata()
lexeme_index = load_lexeme_index()

# Sidebar controls
st.sidebar.header("Search")
query = st.sidebar.text_input("Ruland's entry (supports * wildcard)", value="Mercurius metallorum")

st.sidebar.header("Instances table")
load_full_hits = st.sidebar.toggle(
    "Load full cross-corpus hit file (uses instance_fname)",
    value=True,
)

st.sidebar.caption("Corpus filters (apply to the instances table):")
show_emlap = st.sidebar.checkbox("EMLAP", value=True)
show_nosc = st.sidebar.checkbox("NOSCEMUS", value=True)
show_cc = st.sidebar.checkbox("Corpus Corporum (CC)", value=True)

drop_overlap = st.sidebar.checkbox(
    "Filter out Noscemus hits that overlap with EMLAP (if_noscemus_id)",
    value=True,
)

# Find matches
matches = find_matching_lemmas(query, lexeme_index)

if not matches:
    st.error("No entries found. Try a different search.")
    st.stop()

# Select entry (if multiple)
if len(matches) == 1:
    selected = matches[0]
    selected_lemma = selected["Lemma"]
else:
    st.warning("Multiple matches found. Select one:")
    options = [
        f"{m['Lemma']}  |  EMLAP:{m.get('emlap_instances_N', 0)}  NOS:{m.get('noscemus_instances_N', 0)}  CC:{m.get('cc_instances_N', 0)}"
        for m in matches
    ]
    chosen = st.selectbox("Select an entry", options)
    chosen_lemma = chosen.split("|")[0].strip()
    selected = next(m for m in matches if m["Lemma"] == chosen_lemma)
    selected_lemma = chosen_lemma

# Summary metrics
colA, colB, colC = st.columns(3)
colA.metric("EMLAP instances", int(selected.get("emlap_instances_N", 0)))
colB.metric("NOSCEMUS instances", int(selected.get("noscemus_instances_N", 0)))
colC.metric("CC instances", int(selected.get("cc_instances_N", 0)))

# Load hits (optional)
hits_df = pd.DataFrame()
if load_full_hits:
    fname = str(selected.get("instance_fname", "") or "")
    hits_df = load_hits_df(fname)

    if hits_df.empty:
        st.warning("Hit file could not be loaded (missing / empty / unreadable). Table will be empty.")
else:
    st.info("Turn on “Load full cross-corpus hit file” to browse instance rows across corpora.")

# EMLAP plots (derived from hits if available; otherwise show a message)
st.header("EMLAP distribution (plots)")

if load_full_hits and not hits_df.empty:
    grouped, perwork = compute_emlap_distributions_from_hits(hits_df, emlap_metadata)
    if grouped is None or perwork is None or grouped.empty:
        st.warning("No EMLAP instances for this entry (or could not derive EMLAP work IDs).")
    else:
        fig = plot_emlap_fourpanel(selected_lemma, grouped, perwork)
        st.pyplot(fig)
else:
    st.warning("Plots require loading the full hit file (toggle it on).")

# Instances table
st.header("Instances (table)")

if load_full_hits and not hits_df.empty:
    table_df = filter_hits_for_table(
        hits_df=hits_df,
        show_emlap=show_emlap,
        show_nosc=show_nosc,
        show_cc=show_cc,
        noscemus_emlap_ids=noscemus_emlap_ids,
        drop_emlap_overlap_from_noscemus=drop_overlap,
    )

    if table_df.empty:
        st.info("No rows match the current corpus filters.")
    else:
        # Add some helpful derived columns (safe even if missing)
        table_df = table_df.copy()
        table_df["work_id"] = table_df["grela_id"].astype("string").map(emlap_work_id)

        # Optional: join EMLAP metadata labeldate for nicer browsing
        md_map = emlap_metadata.set_index(emlap_metadata["no."].astype(str))["labeldate"].to_dict()
        table_df["emlap_work"] = table_df["work_id"].map(lambda x: md_map.get(str(x)) if pd.notna(x) else None)

        # Pick a sane column order (keep the rest after)
        preferred = [
            "corpus",
            "grela_id",
            "emlap_work",
            "target_phrase",
            "kwic_text",
            "target_sentence_text",
            "author",
            "title",
            "not_before",
            "not_after",
        ]
        cols = [c for c in preferred if c in table_df.columns] + [c for c in table_df.columns if c not in preferred]
        table_df = table_df[cols]

        st.dataframe(table_df, use_container_width=True, height=450)

        # download
        csv_bytes = table_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered instances as CSV",
            data=csv_bytes,
            file_name=f"{selected_lemma}_instances_filtered.csv",
            mime="text/csv",
        )
else:
    st.write(pd.DataFrame({"empty": []}))


# ---------------------------
# Exploring Contextual Embeddings (optional)
# ---------------------------
st.header("Exploring Contextual Embeddings")

with st.expander("Open: 3D projections + cross-subset similarity", expanded=False):
    st.markdown(
        """
This section samples up to **100 instances per subset**, computes **KWIC target-token embeddings**
(layer 11 of your WSD-optimised Latin BERT), and renders a **3D plot**:
x = `date_random`, y/z = 2D projection of embeddings.

It also computes an average cosine **similarity/distance** matrix by random pairing between subsets.
"""
    )

    # Controls
    c1, c2, c3, c4 = st.columns([1.1, 1.2, 1.1, 1.1])
    with c1:
        form = st.radio("Figure type", ["static", "interactive"], horizontal=True)
    with c2:
        projection = st.selectbox("Projection", list(iee.REDUCTIONS.keys()), index=list(iee.REDUCTIONS.keys()).index("pca10_tsne"))
    with c3:
        kind = st.radio("Cosine metric", ["similarity", "distance"], horizontal=True)
    with c4:
        n_pairs = st.number_input("Pairs per cell", min_value=10, max_value=5000, value=100, step=50)

    seed = st.number_input("Random seed", min_value=0, max_value=1_000_000, value=42, step=1)

    run = st.button("Generate embedding plot", type="primary")

    if run:
        fname = str(selected.get("instance_fname", "") or "")
        if not fname:
            st.error("No instance_fname for this lemma (cannot load hit file).")
            st.stop()

        # Cache-heavy step: load -> enrich -> sample -> embed
        @st.cache_data(show_spinner=True)
        def _compute_sample(fname: str):
            hits = load_hits_df(fname)
            full, full_sizes = iee.enrich_instances(hits)
            sample, sample_sizes = iee.produce_and_enrich_samples(full)
            return sample, full_sizes, sample_sizes

        instances_sample, full_sizes, sample_sizes = _compute_sample(fname)

        # Similarity matrix (light-ish; depends on n_pairs)
        S = iee.avg_cosine_matrix_random_pairs(
            instances_sample,
            order=["emlap", "noscemus_alchymia", "noscemus_rest", "cc"],
            n_pairs=int(n_pairs),
            random_state=int(seed),
            kind=kind,
        )

        # Plot (uses your module’s geometry and hover)
        fig = iee.make_plot(
            instances_sample,
            iee.palette,
            full_sizes,
            sample_sizes,
            projection=projection,
            form=form,
        )

        st.subheader("3D projection")
        if form == "static":
            st.pyplot(fig, clear_figure=True)
        else:
            st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"Average cosine {kind} matrix (random pairing, n_pairs={int(n_pairs)})")
        st.dataframe(S, use_container_width=True)

        st.subheader("Sampled instances used")
        st.caption("This is the sampled/enriched DataFrame (up to 100 per subset).")
        st.dataframe(
            instances_sample.drop(columns=["embedding"], errors="ignore"),
            use_container_width=True,
            height=320
        )