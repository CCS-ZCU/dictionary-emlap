import embedding_helpers

import pickle

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import os
import numpy as np
import json
import importlib
from pathlib import Path
import requests
from transformers import AutoModel, AutoTokenizer
from transformers import PreTrainedTokenizer
from tensor2tensor.data_generators import text_encoder
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold      import TSNE
import umap
import tempun

import plotly.graph_objects as go
importlib.reload(embedding_helpers)
from embedding_helpers import *

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
from matplotlib.lines import Line2D

lexeme_df = pd.read_parquet("../data/large_files/ruland-emlap-grela.parquet")


outdir = Path("../data/large_files/emlap_ruland_instances/")

noscemus_alchemy_ids = pd.read_pickle("../data/noscemus_alchemy_ids.pickle")

lemma_fname_dict = dict(zip(lexeme_df["Lemma"], lexeme_df["instance_fname"]))


base_path = "/srv/models/latin-bert"

# Initialize the tokenizer with the vocab.txt file and the encoder
vocab_file_path = "/srv/models/latin-bert/vocab.txt" # "/Users/vojtechkase/Projects/latin-bert/models/latin_bert/vocab.txt"  # Update this path
subword_tokenizer_path = "/srv/models/latin-bert/latin.subword.encoder"
# Update this path
encoder = text_encoder.SubwordTextEncoder(subword_tokenizer_path)

spec = importlib.util.spec_from_file_location("latin_tokenizer", os.path.join(base_path, "latin_tokenizer.py"))
latin_tokenizer = importlib.util.module_from_spec(spec)
loader = importlib.util.LazyLoader(spec.loader)
spec.loader.exec_module(latin_tokenizer)

LatinTokenizer = latin_tokenizer.LatinTokenizer

tokenizer_labert = LatinTokenizer(vocab_file_path, encoder)
# model_labert = AutoModel.from_pretrained(base_path)

model_labert_wsd = AutoModel.from_pretrained(
    "../../labyrinth/data/models/latin-bert-wic-ft",
    output_hidden_states=True,
    output_attentions=False
)


def read_hits(fname):
    try:
        path = os.path.join(outdir, fname)
        instances = pd.read_parquet(path).to_dict("records")
    except:
        instances = []
    instances = pd.DataFrame(instances_list)
    return instances



def get_subset(row):
    if row["grela_source"] == "noscemus":
        if row["grela_id"] in noscemus_alchemy_ids.tolist():
            return "noscemus_alchymia"
        else:
            return "noscemus_rest"
    else:
        return row["grela_source"]

def enrich_instances(instances):
    instances = instances[instances["grela_id"] != "noscemus_668514"].copy()  # exclude Ruland
    instances = instances[~instances["grela_id"].isin(['noscemus_713324', 'noscemus_26765', 'noscemus_597737', 'noscemus_971293', 'noscemus_688159'])].copy()  # exclude Noscemus works appearing in Emlap



    instances.loc[:, "grela_source"] = instances["grela_id"].astype(str).str.partition("_")[0]
    instances = instances[instances["grela_source"] != "vulgate"].copy()

    instances.loc[:, "subsets"] = instances.apply(get_subset, axis=1)
    instances_full_sizes_dict = instances.value_counts("subsets").to_dict()
    return instances, instances_full_sizes_dict



def produce_and_enrich_samples(instances):
    instances = (
        instances
        .sample(frac=1, random_state=42)      # shuffle rows
        .groupby("subsets", group_keys=False) # then take first up to 100 per group
        .head(100)
        .reset_index(drop=True)
    )

    # obtain random date
    instances.loc[:, "date_random"] = [
        (d[0] if isinstance(d, (list, tuple, np.ndarray)) else d)
        for d in instances.apply(
            lambda row: tempun.model_date(row["not_before"], row["not_after"]),
            axis=1
        )
    ]

    # compute embedding
    instances.loc[:, "embedding"] = instances.apply(
        lambda row: embedding_helpers.embed_emlap_instance(
            instance=row,
            tokenizer=tokenizer_labert,
            model=model_labert_wsd,
            device="cuda",
            layer_idx=11,
            piece_pooling="mean",
            context_lemmatized=True,
            target_lemmatized=True,
            context_pos_filtered=True,
        )["embedding"],
        axis=1
    )

    instances_samples_sizes_dict = instances["subsets"].value_counts().to_dict()
    return instances, instances_samples_sizes_dict

def _safe_tsne_perplexity(n_samples: int, requested: float) -> float:
    # sklearn requires perplexity < n_samples; and perplexity should be >= 1
    if n_samples <= 1:
        return 1.0
    return float(min(requested, max(1.0, n_samples - 1.0)))

def pca10_tsne(X, random_state=42, perplexity=30):
    X = np.asarray(X)
    n = X.shape[0]
    perp = _safe_tsne_perplexity(n, perplexity)
    X10 = PCA(n_components=min(10, X.shape[1], n), random_state=random_state).fit_transform(X)
    return TSNE(n_components=2, perplexity=perp, init="pca", random_state=random_state).fit_transform(X10)

REDUCTIONS = {
    # --- UMAP variants ---
    "umap_10_0.05": lambda: umap.UMAP(n_neighbors=10, min_dist=0.05,
                                      metric="cosine", random_state=42),
    "umap_50_0.5": lambda: umap.UMAP(n_neighbors=50, min_dist=0.5,
                                     metric="cosine", random_state=42),
    "umap_100_0.8": lambda: umap.UMAP(n_neighbors=100, min_dist=0.8,
                                      metric="cosine", random_state=42),
    # --- t-SNE variants ---
    "tsne_10": lambda: TSNE(n_components=2, perplexity=10,
                            metric="cosine", init="pca",
                            learning_rate=200, random_state=42),
    "tsne_30": lambda: TSNE(n_components=2, perplexity=30,
                            metric="cosine", init="pca",
                            learning_rate=200, random_state=42),
    "tsne_50": lambda: TSNE(n_components=2, perplexity=50,
                            metric="cosine", init="pca",
                            learning_rate=200, random_state=42),
    # --- PCA variants ---
    "pca_std": lambda: PCA(n_components=2, whiten=False, random_state=42),
    "pca_whiten": lambda: PCA(n_components=2, whiten=True, random_state=42),
    # --- PCA→t-SNE speed trick ---
    "pca10_tsne": lambda: (lambda X: pca10_tsne(X, random_state=42, perplexity=30))
}


def avg_cosine_matrix_random_pairs(
    instances,
    subset_col="subsets",
    emb_col="embedding",
    order=None,
    n_pairs=100,
    random_state=42,
    kind="distance",
):
    if kind not in {"distance", "similarity"}:
        raise ValueError("kind must be 'distance' or 'similarity'")

    rng = np.random.default_rng(random_state)

    if order is None:
        order = list(pd.Index(instances[subset_col].dropna().unique()).sort_values())

    mats = {}
    sizes = {}

    for s in order:
        ser = instances.loc[instances[subset_col] == s, emb_col]

        # drop missing embeddings explicitly
        ser = ser.dropna()

        if len(ser) == 0:
            mats[s] = np.empty((0, 0), dtype=np.float32)
            sizes[s] = 0
            continue

        X = np.vstack(ser.apply(lambda v: np.asarray(v, dtype=np.float32)).to_numpy())
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        mats[s] = X
        sizes[s] = X.shape[0]

    out = pd.DataFrame(index=order, columns=order, dtype=float)

    for i, a in enumerate(order):
        Xa = mats[a]
        na = sizes[a]

        for j, b in enumerate(order):
            if j < i:
                continue

            Xb = mats[b]
            nb = sizes[b]

            if na == 0 or nb == 0:
                mean_sim = np.nan

            elif a != b:
                ia = rng.integers(0, na, size=n_pairs)
                ib = rng.integers(0, nb, size=n_pairs)
                sims = np.einsum("ij,ij->i", Xa[ia], Xb[ib])
                mean_sim = float(sims.mean())

            else:
                if na < 2:
                    mean_sim = np.nan
                else:
                    ia = rng.integers(0, na, size=n_pairs)
                    ib = rng.integers(0, na, size=n_pairs)
                    same = (ia == ib)
                    while same.any():
                        ib[same] = rng.integers(0, na, size=int(same.sum()))
                        same = (ia == ib)
                    sims = np.einsum("ij,ij->i", Xa[ia], Xa[ib])
                    mean_sim = float(sims.mean())

            val = (1.0 - mean_sim) if kind == "distance" else mean_sim
            out.loc[a, b] = val
            out.loc[b, a] = val

    return out

def get_2d(emb, projection_factory, random_state=42):
    """
    projection_factory: callable that returns either
      - an sklearn object with fit_transform
      - or a callable X -> 2D array (your pca10_tsne trick)
    This function makes TSNE perplexity valid for tiny n, and falls back to PCA2 when needed.
    """
    X = np.asarray(emb)
    n = X.shape[0]

    # tiny sample: no projection method is meaningful, but we must not crash
    if n < 3:
        Y = PCA(n_components=2, random_state=random_state).fit_transform(X)
        return Y[:, 0], Y[:, 1]

    reducer = projection_factory()

    # --- Case A: "normal" sklearn reducers ---
    if hasattr(reducer, "fit_transform"):
        # If it’s TSNE, adjust perplexity in-place
        if isinstance(reducer, TSNE):
            reducer.set_params(perplexity=_safe_tsne_perplexity(n, reducer.perplexity))

        try:
            Y = reducer.fit_transform(X)
        except ValueError:
            # last-resort fallback (e.g., TSNE still unhappy, PCA dims, etc.)
            Y = PCA(n_components=2, random_state=random_state).fit_transform(X)

        return Y[:, 0], Y[:, 1]

    # --- Case B: callable reducers (like pca10_tsne) ---
    # We can't introspect its internal TSNE params reliably, so we just do the safe version here.
    try:
        Y = reducer(X)
    except ValueError as e:
        # handle the exact failure you saw: perplexity must be less than n_samples
        # => re-run with a safe pca->tsne pipeline using requested "30" as default
        perp = _safe_tsne_perplexity(n, 30)
        X10 = PCA(n_components=min(10, X.shape[1], n), random_state=random_state).fit_transform(X)
        Y = TSNE(n_components=2, perplexity=perp, init="pca", random_state=random_state).fit_transform(X10)

    return Y[:, 0], Y[:, 1]

palette = {"cc": "orange",
           "noscemus_rest": "green",
           "noscemus_alchymia": "blue",
           "emlap": "red"
           }




def clean(x):
    return "" if pd.isna(x) else str(x)

def produce_hover(row):
    return "'{}' ({}, {}, [{}-{}])".format(
        clean(row.get("kwic_text")),
        clean(row.get("author")),
        clean(row.get("title")),
        clean(row.get("not_before")),
        clean(row.get("not_after")),
    )

def present_order(instances, palette, preferred=("emlap","noscemus_alchymia","noscemus_rest","cc")):
    present = set(instances["subsets"].dropna().unique())
    order = [s for s in preferred if s in present]
    # if there are any unexpected subsets, append them (keeps app robust)
    order += [s for s in sorted(present) if s not in order]
    # fallback: if somehow empty, return []
    return order

# ------------------------------------------------------------------

def make_plot(instances, palette, instances_full_sizes_dict, instances_samples_sizes_dict, projection, form):
    instances["hover"] = instances.apply(lambda row: produce_hover(row), axis=1)

    order = present_order(instances, palette)  # <-- only present ones

    labels = []
    colors_for_legend = []
    for subset in order:
        labels.append("{0} ({1} / {2})".format(
            subset,
            instances_samples_sizes_dict.get(subset, 0),
            instances_full_sizes_dict.get(subset, 0)
        ))
        colors_for_legend.append(palette.get(subset, "gray"))

    emb = np.stack(instances["embedding"])
    label = "l11_labert_wsd"
    xs = instances["date_random"]
    ys, zs = get_2d(emb, REDUCTIONS[projection])
    colors = instances["subsets"].map(palette)

    if form == "static":
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs, ys, zs, c=colors, s=15, alpha=0.7)

        ax.set_title(f"3D projection of {label}", fontsize=14)
        ax.set_ylabel("Component 1")
        ax.set_zlabel("Component 2")
        ax.set_xlabel("Date")
        ax.set_xlim(min(xs) - 20, max(xs) + 20)

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=lbl,
                   markerfacecolor=col, markersize=8)
            for lbl, col in zip(labels, colors_for_legend)
        ]
        ax.legend(handles=legend_elements, title="Subsets (sample / full)",
                  bbox_to_anchor=(0.85, 1), loc='upper left')

        plt.tight_layout()
        return fig

    else:
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode="markers",
                    marker=dict(size=4, opacity=0.7, color=colors),
                    text=instances["hover"],
                    hovertemplate="%{text}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title=f"3D projection of {label}",
            scene=dict(
                xaxis_title="Date",
                yaxis_title="Projected dimension Y",
                zaxis_title="Projected dimension Z",
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        return fig

def from_lemma_to_plot(lemma, projection, form, n_pairs=100, kind="similarity"):
    fname = lemma_fname_dict[lemma]
    instances_full = read_hits(fname)
    instances_full, instances_full_sizes_dict = enrich_instances(instances_full)
    instances_sample, instances_samples_sizes_dict = produce_and_enrich_samples(instances_full)

    order = present_order(instances_sample, palette)
    S = avg_cosine_matrix_random_pairs(
        instances_sample,
        order=order,
        n_pairs=n_pairs,
        kind=kind,
    )

    fig = make_plot(
        instances_sample, palette,
        instances_full_sizes_dict, instances_samples_sizes_dict,
        projection, form
    )
    return fig, fname, instances_sample, S, instances_full_sizes_dict, instances_samples_sizes_dict

def sample_embed_plot_from_instances(
    instances_enriched: pd.DataFrame,
    projection: str,
    form: str,
    *,
    n_pairs: int = 100,
    kind: str = "similarity",
    random_state: int = 42,
    max_per_subset: int = 100,
):
    """
    Takes *already enriched* instances (i.e. output of enrich_instances),
    then:
      - samples up to max_per_subset per subset
      - adds date_random + embeddings
      - computes similarity/distance matrix across *present* subsets only
      - builds plot (static mpl or interactive plotly)

    Returns:
      fig, instances_sample, S, instances_full_sizes_dict, instances_samples_sizes_dict
    """
    if instances_enriched is None or len(instances_enriched) == 0:
        empty = pd.DataFrame()
        return None, empty, empty, {}, {}

    # counts in the *full enriched* set (used for legend)
    instances_full_sizes_dict = instances_enriched.value_counts("subsets").to_dict()

    # sample deterministically per subset
    instances_sample = (
        instances_enriched
        .sample(frac=1, random_state=int(random_state))
        .groupby("subsets", group_keys=False)
        .head(int(max_per_subset))
        .reset_index(drop=True)
    )

    # add date_random + embeddings (your existing routine)
    # note: produce_and_enrich_samples currently hardcodes head(100) and random_state=42
    # we want to respect random_state/max_per_subset, so we inline the safe bits:
    instances_sample.loc[:, "date_random"] = [
        (d[0] if isinstance(d, (list, tuple, np.ndarray)) else d)
        for d in instances_sample.apply(
            lambda row: tempun.model_date(row["not_before"], row["not_after"]),
            axis=1
        )
    ]

    instances_sample.loc[:, "embedding"] = instances_sample.apply(
        lambda row: embedding_helpers.embed_emlap_instance(
            instance=row,
            tokenizer=tokenizer_labert,
            model=model_labert_wsd,
            device="cuda",
            layer_idx=11,
            piece_pooling="mean",
            context_lemmatized=True,
            target_lemmatized=True,
            context_pos_filtered=True,
        )["embedding"],
        axis=1
    )

    instances_samples_sizes_dict = instances_sample["subsets"].value_counts().to_dict()

    # present subset order only (robust for missing subsets)
    order = present_order(instances_sample, palette)

    S = avg_cosine_matrix_random_pairs(
        instances_sample,
        order=order,
        n_pairs=int(n_pairs),
        random_state=int(random_state),
        kind=kind,
    )

    fig = make_plot(
        instances_sample,
        palette,
        instances_full_sizes_dict,
        instances_samples_sizes_dict,
        projection=projection,
        form=form,
    )

    return fig, instances_sample, S, instances_full_sizes_dict, instances_samples_sizes_dict