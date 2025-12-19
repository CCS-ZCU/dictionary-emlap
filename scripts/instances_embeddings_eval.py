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
    "pca10_tsne": lambda: (
        lambda X: TSNE(n_components=2, perplexity=30,
                       init="pca", random_state=42)
        .fit_transform(PCA(n_components=10, random_state=42)
                       .fit_transform(X))
    )
}


def avg_cosine_matrix_random_pairs(
    instances,
    subset_col="subsets",
    emb_col="embedding",
    order=None,
    n_pairs=100,
    random_state=42,
    kind="distance",   # "distance" (1-cos) or "similarity" (cos)
):
    if kind not in {"distance", "similarity"}:
        raise ValueError("kind must be 'distance' or 'similarity'")

    rng = np.random.default_rng(random_state)

    if order is None:
        order = list(pd.Index(instances[subset_col].dropna().unique()).sort_values())

    # collect normalized embedding matrices per subset
    mats = {}
    for s in order:
        X = np.vstack(
            instances.loc[instances[subset_col] == s, emb_col]
            .apply(lambda v: np.asarray(v, dtype=np.float32))
            .to_numpy()
        )
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)  # L2 normalize
        mats[s] = X

    out = pd.DataFrame(index=order, columns=order, dtype=float)

    for i, a in enumerate(order):
        Xa = mats[a]
        na = Xa.shape[0]

        for j, b in enumerate(order):
            if j < i:
                continue

            Xb = mats[b]
            nb = Xb.shape[0]

            if na == 0 or nb == 0:
                mean_sim = np.nan

            elif a != b:
                ia = rng.integers(0, na, size=n_pairs)
                ib = rng.integers(0, nb, size=n_pairs)
                sims = np.einsum("ij,ij->i", Xa[ia], Xb[ib])  # cosine similarity
                mean_sim = float(sims.mean())

            else:
                # within-subset: pair two *different* items each time
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
            out.loc[b, a] = val  # symmetry

    return out

def get_2d(emb, projection):
    reducer = projection()
    if hasattr(reducer, "fit_transform"):
        embedding_2d = reducer.fit_transform(emb)
    else:  # callable expecting X directly (pca10_tsne)
        embedding_2d = reducer(emb)
    return embedding_2d[:, 0], embedding_2d[:, 1]

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
# ------------------------------------------------------------------

# Embeddings and labels
def make_plot(instances, palette, instances_full_sizes_dict, instances_samples_sizes_dict, projection, form):
    instances["hover"] = instances.apply(lambda row: produce_hover(row), axis=1)
    hovers = instances["hover"]
    labels = []
    for subset in palette.keys():
        extended_label = "{0} ({1} / {2})".format(subset, instances_samples_sizes_dict.get(subset, 0), instances_full_sizes_dict.get(subset, 0))
        labels.append(extended_label)
    emb = np.stack(instances["embedding"])
    label = "l11_labert_wsd"
    xs = instances["date_random"]
    ys, zs = get_2d(emb, REDUCTIONS[projection])
    colors = instances["subsets"].map(palette)
    if form == "static":
    # 3D scatter plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(xs, ys, zs, c=colors, s=15, alpha=0.7)

        ax.set_title(f"3D projection of {label}", fontsize=14)
        ax.set_ylabel("Component 1")
        ax.set_zlabel("Component 2")
        ax.set_xlabel("Date")
        ax.set_xlim(min(xs) - 20, max(xs) + 20)

        # Legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=str(lbl),
                   markerfacecolor=color, markersize=8) for lbl, color in zip(labels, palette.values())
        ]
        ax.legend(handles=legend_elements, title="Subsets (sample / full)", bbox_to_anchor=(0.85, 1), loc='upper left')

        plt.tight_layout()
        return fig
    else:
        fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
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

def from_lemma_to_plot(lemma, projection, form):
    fname = lemma_fname_dict[lemma]
    instances_full = read_hits(fname)
    instances_full, instances_full_sizes_dict = enrich_instances(instances_full)
    instances_sample, instances_samples_sizes_dict = produce_and_enrich_samples(instances_full)
    S = avg_cosine_matrix_random_pairs(instances_sample, order=["emlap","noscemus_alchymia","noscemus_rest","cc"],
                                   n_pairs=100, kind="similarity")
    fig = make_plot(instances_sample, palette, instances_full_sizes_dict, instances_samples_sizes_dict, projection, form)
    return fig, fname, instances_sample, S, instances_full_sizes_dict, instances_samples_sizes_dict