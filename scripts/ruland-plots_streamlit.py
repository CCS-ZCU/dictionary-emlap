# ruland-plots_streamlit.py
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
HITS_DIR = Path("../data/large_files/emlap_ruland_instances/")  # per-entry parquet hit files


# ---------------------------
# Cached loaders
# ---------------------------
@st.cache_data(show_spinner=True, ttl=24*3600)
def load_emlap_metadata():
    emlap_metadata = pd.read_csv(
        "https://raw.githubusercontent.com/CCS-ZCU/EMLAP_ETL/refs/heads/master/data/emlap_metadata.csv",
        sep=";",
    )

    emlap_metadata["labeldate"] = emlap_metadata.apply(
        lambda row: f"{row['working_title']} ({row['date_publication']})",
        axis=1,
    )

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

    keep = ["Lemma", "instance_fname", "emlap_instances_N", "noscemus_instances_N", "cc_instances_N"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].drop_duplicates(subset=["Lemma"]).copy()

    df["Lemma"] = df["Lemma"].astype("string")
    if "instance_fname" in df.columns:
        df["instance_fname"] = df["instance_fname"].astype("string")

    for c in ["emlap_instances_N", "noscemus_instances_N", "cc_instances_N"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    return df


import pyarrow.parquet as pq

@st.cache_data(show_spinner=True)
def load_hits_df(fname: str, columns=None) -> pd.DataFrame:
    """Load hit file for an entry. Robust to missing columns."""
    if not fname:
        return pd.DataFrame()

    # guard against pandas string NA values leaking into filenames
    fname_str = str(fname)
    if fname_str.lower() in {"<na>", "nan", "none"}:
        return pd.DataFrame()

    path = HITS_DIR / fname_str
    if not path.exists():
        return pd.DataFrame()

    try:
        if columns is None:
            return pd.read_parquet(path)

        # make hashable/stable for cache + sanitize
        columns = list(columns)

        # read schema without loading full parquet
        avail = pq.ParquetFile(path).schema.names
        cols = [c for c in columns if c in avail]

        # if none of the requested cols exist, fall back to reading everything
        if not cols:
            return pd.read_parquet(path)

        return pd.read_parquet(path, columns=cols)

    except Exception:
        # last-resort fallback: try loading full file
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()

# ---------------------------
# Helpers
# ---------------------------
def _extract_after_prefix(grela_id: str, prefix: str) -> str | None:
    if not isinstance(grela_id, str):
        return None
    m = re.match(rf"^{re.escape(prefix)}[:_](.+)$", grela_id)
    return m.group(1) if m else None


def emlap_work_id(grela_id: str) -> str | None:
    return _extract_after_prefix(grela_id, "emlap")


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
# Plotting from enriched instances
# ---------------------------
def compute_emlap_distributions_from_instances(instances_enriched: pd.DataFrame, emlap_metadata: pd.DataFrame):
    """
    Same as before, but the input is the *enriched/filtered* instances df.
    We use only subset == 'emlap', so the exclusions/overlap logic is aligned with embeddings.
    """
    if instances_enriched.empty or "grela_id" not in instances_enriched.columns:
        return None, None

    if "subsets" not in instances_enriched.columns:
        return None, None

    em = instances_enriched.loc[instances_enriched["subsets"] == "emlap"].copy()
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

    ax1 = axes[0, 0]
    ax1.bar(grouped["Bidecade"], grouped["Frequency"], color="darkblue")
    ax1.set_xlabel("Bidecades", fontsize=11)
    ax1.set_ylabel("Absolute Frequency", fontsize=11)
    ax1.set_title(f"(A) Absolute Frequency of '{entry}' by Bidecades", fontsize=13)
    ax1.set_xticks(range(len(grouped["Bidecade"])))
    ax1.set_xticklabels(grouped["Bidecade"], rotation=90, fontsize=9)

    ax2 = axes[0, 1]
    ax2.bar(grouped["Bidecade"], grouped["RelativeFrequency"], color="darkgreen")
    ax2.set_xlabel("Bidecades", fontsize=11)
    ax2.set_ylabel("Relative Frequency (Frequency / tokens_N)", fontsize=11)
    ax2.set_title(f"(B) Relative Frequency of '{entry}' by Bidecades", fontsize=13)
    ax2.set_xticks(range(len(grouped["Bidecade"])))
    ax2.set_xticklabels(grouped["Bidecade"], rotation=90, fontsize=9)

    ax3 = axes[1, 0]
    ax3.bar(perwork["labeldate"], perwork["Frequency"], color="darkblue")
    ax3.set_xlabel("Works", fontsize=11)
    ax3.set_ylabel("Absolute Frequency", fontsize=11)
    ax3.set_title(f"(C) Absolute Frequency of '{entry}' by Works", fontsize=13)
    ax3.set_xticks(range(len(perwork["labeldate"])))
    ax3.set_xticklabels(perwork["labeldate"], rotation=90, fontsize=8)

    ax4 = axes[1, 1]
    ax4.bar(perwork["labeldate"], perwork["RelativeFrequency"], color="darkgreen")
    ax4.set_xlabel("Works", fontsize=11)
    ax4.set_ylabel("(D) Relative Frequency (Frequency / tokens_N)", fontsize=11)
    ax4.set_title(f"Relative Frequency of '{entry}' by Works", fontsize=13)
    ax4.set_xticks(range(len(perwork["labeldate"])))
    ax4.set_xticklabels(perwork["labeldate"], rotation=90, fontsize=8)

    plt.tight_layout()
    return fig


def truncate_series(s: pd.Series, n: int) -> pd.Series:
    s = s.astype("string")
    too_long = s.str.len().fillna(0) > n
    return s.where(~too_long, s.str.slice(0, n) + "…")


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

emlap_metadata, _ = load_emlap_metadata()
lexeme_index = load_lexeme_index()

# Sidebar controls
st.sidebar.header("Search")
query = st.sidebar.text_input("Ruland's entry (supports * wildcard)", value="Mercurius metallorum")

st.sidebar.header("Instances table")
load_full_hits = st.sidebar.toggle(
    "Load full cross-corpus hit file (uses instance_fname)",
    value=True,
)

st.sidebar.caption("Subset filters (apply to the instances table):")
show_emlap = st.sidebar.checkbox("emlap", value=True)
show_nosc_alch = st.sidebar.checkbox("noscemus_alchymia", value=True)
show_nosc_rest = st.sidebar.checkbox("noscemus_rest", value=True)
show_cc = st.sidebar.checkbox("cc", value=True)

page_size = st.sidebar.number_input("Table page size", 100, 5000, 1000, 100)
truncate_chars = st.sidebar.number_input("Truncate text columns to chars", 50, 2000, 300, 50)

# Find matches
matches = find_matching_lemmas(query, lexeme_index)
if not matches:
    st.error("No entries found. Try a different search.")
    st.stop()

# Select entry
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

fname = str(selected.get("instance_fname", "") or "")

# ---------------------------
# Load + enrich ONCE (for plots + table)
# ---------------------------
instances_enriched = pd.DataFrame()
instances_full_sizes_dict = {}

if load_full_hits:
    # For plots+table we can load a limited set of columns (fast + avoids huge msg sizes).
    include_sentence = st.sidebar.checkbox("Include full sentence text in table load", value=True)

    base_cols = [
        "grela_id",
        "target_phrase",
        "kwic_text",
        "author",
        "title",
        "not_before",
        "not_after",
    ]
    if include_sentence:
        base_cols.append("target_sentence_text")

    hits_df = load_hits_df(fname, columns=base_cols) if fname else pd.DataFrame()

    if hits_df.empty:
        st.warning("Hit file could not be loaded (missing / empty / unreadable).")
    else:
        instances_enriched, instances_full_sizes_dict = iee.enrich_instances(hits_df)

else:
    st.info("Turn on “Load full cross-corpus hit file” to browse instance rows across corpora and show plots.")


# ---------------------------
# Summary metrics (aligned with enriched subsets)
# ---------------------------
colA, colB, colC, colD = st.columns(4)
colA.metric("emlap (filtered)", int(instances_full_sizes_dict.get("emlap", 0)))
colB.metric("noscemus_alchymia (filtered)", int(instances_full_sizes_dict.get("noscemus_alchymia", 0)))
colC.metric("noscemus_rest (filtered)", int(instances_full_sizes_dict.get("noscemus_rest", 0)))
colD.metric("cc (filtered)", int(instances_full_sizes_dict.get("cc", 0)))


# ---------------------------
# EMLAP distribution (plots) — derived from enriched df
# ---------------------------
st.header("EMLAP distribution (plots)")

if load_full_hits and not instances_enriched.empty:
    grouped, perwork = compute_emlap_distributions_from_instances(instances_enriched, emlap_metadata)
    if grouped is None or perwork is None or grouped.empty:
        st.warning("No (filtered) EMLAP instances for this entry (or could not derive EMLAP work IDs).")
    else:
        fig = plot_emlap_fourpanel(selected_lemma, grouped, perwork)
        st.pyplot(fig)
else:
    st.warning("Plots require loading the hit file (toggle it on).")


# ---------------------------
# Instances table (paginated) — uses enriched df + same subsets
# ---------------------------
st.header("Instances (table)")

if load_full_hits and not instances_enriched.empty:
    allowed = set()
    if show_emlap:
        allowed.add("emlap")
    if show_nosc_alch:
        allowed.add("noscemus_alchymia")
    if show_nosc_rest:
        allowed.add("noscemus_rest")
    if show_cc:
        allowed.add("cc")

    if not allowed:
        st.info("No subsets selected.")
    else:
        table_df = instances_enriched.loc[instances_enriched["subsets"].isin(allowed)].copy()

        if table_df.empty:
            st.info("No rows match the current subset filters.")
        else:
            table_df["work_id"] = table_df["grela_id"].astype("string").map(emlap_work_id)

            md_map = emlap_metadata.set_index(emlap_metadata["no."].astype(str))["labeldate"].to_dict()
            table_df["emlap_work"] = table_df["work_id"].map(lambda x: md_map.get(str(x)) if pd.notna(x) else None)

            preferred = [
                "subsets",
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

            display_cols = [
                "subsets", "grela_id", "emlap_work",
                "target_phrase", "kwic_text", "target_sentence_text",
                "author", "title", "not_before", "not_after",
            ]
            display_cols = [c for c in display_cols if c in table_df.columns]
            display_df = table_df[display_cols].copy()

            for c in ["kwic_text", "target_sentence_text", "target_phrase", "author", "title"]:
                if c in display_df.columns:
                    display_df[c] = truncate_series(display_df[c], int(truncate_chars))

            total_rows = len(display_df)
            total_pages = max(1, (total_rows - 1) // int(page_size) + 1)

            page = st.sidebar.number_input(
                "Table page (0 = first)",
                min_value=0,
                max_value=int(total_pages - 1),
                value=0,
                step=1,
            )
            start = int(page) * int(page_size)
            end = min(start + int(page_size), total_rows)

            st.caption(
                f"Showing rows {start:,}–{end:,} of {total_rows:,} "
                f"(page {int(page)+1}/{total_pages}, truncated for display)."
            )
            st.dataframe(display_df.iloc[start:end], use_container_width=True, height=450)

            # Safer default: download current page only (avoids huge memory spikes)
            page_csv = table_df.iloc[start:end].to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download THIS PAGE as CSV",
                data=page_csv,
                file_name=f"{selected_lemma}_instances_page{int(page)}.csv",
                mime="text/csv",
            )

            with st.expander("Advanced: download ALL filtered instances (may be large)"):
                if st.button("Prepare full CSV (can be heavy)"):
                    full_csv = table_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download ALL filtered instances as CSV",
                        data=full_csv,
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

**Important:** embeddings are computed from the same *filtered/enriched* instances as the table/plots.
"""
    )

    c1, c2, c3, c4 = st.columns([1.1, 1.2, 1.1, 1.1])
    with c1:
        form = st.radio("Figure type", ["static", "interactive"], horizontal=True)
    with c2:
        projection = st.selectbox(
            "Projection",
            list(iee.REDUCTIONS.keys()),
            index=list(iee.REDUCTIONS.keys()).index("pca10_tsne") if "pca10_tsne" in iee.REDUCTIONS else 0,
        )
    with c3:
        kind = st.radio("Cosine metric", ["similarity", "distance"], horizontal=True)
    with c4:
        n_pairs = st.number_input("Pairs per cell", min_value=10, max_value=5000, value=100, step=50)

    seed = st.number_input("Random seed", min_value=0, max_value=1_000_000, value=42, step=1)

    run = st.button("Generate embedding plot", type="primary")

    if run:
        if not fname:
            st.error("No instance_fname for this lemma (cannot load hit file).")
            st.stop()

        # Load FULL hit file for embeddings (needs columns beyond base_cols)
        hits_full = load_hits_df(fname, columns=None)
        if hits_full.empty:
            st.error("Full hit file could not be loaded for embeddings.")
            st.stop()

        # Apply the SAME filtering/enrichment logic as everywhere else
        full_enriched, full_sizes = iee.enrich_instances(hits_full)

        if full_enriched.empty:
            st.warning("No instances left after filtering/enrichment.")
            st.stop()

        # Sample -> embed (max ~100 per subset)
        instances_sample, sample_sizes = iee.produce_and_enrich_samples(full_enriched)

        if instances_sample.empty:
            st.warning("Sampling produced no instances.")
            st.stop()

        order = iee.present_order(instances_sample, iee.palette)

        S = iee.avg_cosine_matrix_random_pairs(
            instances_sample,
            order=order,
            n_pairs=int(n_pairs),
            random_state=int(seed),
            kind=kind,
        )

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
        st.dataframe(
            instances_sample.drop(columns=["embedding"], errors="ignore"),
            use_container_width=True,
            height=320,
        )