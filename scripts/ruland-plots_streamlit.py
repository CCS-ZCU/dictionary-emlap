# Streamlit App for Visualizing Ruland's Entries
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import streamlit as st

# Title and description for the app
st.title("Ruland in Usage")
st.markdown(
    """
    This app visualizes the distribution of instances from Ruland's entries by bidecades and works from the EMLAP corpus. It gives you both absolute and relative values.

    The source code for the app is available from [here](https://github.com/CCS-ZCU/dictionary-emlap).

    Enter an entry (e.g., **Mercurius metallorum**) to see the plots.
    """
)


# Load data files
@st.cache_data
def load_data():
    # Load entries JSON file
    entries_df = pd.read_json("../data/large_files/ruland-emlap.json")

    # Load metadata from the web
    emlap_metadata = pd.read_csv(
        "https://raw.githubusercontent.com/CCS-ZCU/EMLAP_ETL/refs/heads/master/data/emlap_metadata.csv",
        sep=";",
    )

    # Add a labeldate column to metadata
    emlap_metadata["labeldate"] = emlap_metadata.apply(
        lambda row: row["working_title"] + " ({})".format(str(row["date_publication"])),
        axis=1
    )

    # Sort metadata by the date of publication
    emlap_metadata.sort_values("date_publication", ascending=True, inplace=True)

    return entries_df, emlap_metadata


# Load the data
entries_df, emlap_metadata = load_data()

# Create a dropdown for the user to select a lemma
available_keys = entries_df["Lemma"].unique()
entry = st.text_input("Ruland's entry", value="Mercurius metallorum")


# Function to generate the plots
def make_plot_bar(entry):
    # Check if entry exists in the available keys
    if entry not in available_keys:
        st.error("This entry is not in the dictionary.")
        return

    # Get instances IDs for the entry
    instances_ids = entries_df[entries_df["Lemma"] == entry]["instances_ids"].tolist()[0]
    counter = Counter(instances_ids)

    # Extract the keys (labels) and their corresponding counts (frequencies)
    emlap_metadata_instances = emlap_metadata.copy()
    emlap_metadata_instances["Frequency"] = (
        emlap_metadata_instances["No."].map(counter).fillna(0).astype(int)
    )

    # Add bidecade labels to the DataFrame
    def get_bidecade_label(year):
        if not np.isnan(year):  # Handle NaN years safely
            start = (year // 20) * 20  # Determine the starting year of the bidecade
            end = start + 19  # Determine the ending year of the bidecade
            return f"{start}-{end}"
        return None

    emlap_metadata_instances["Bidecade"] = emlap_metadata_instances["date_publication"].apply(get_bidecade_label)

    # Group by Bidecade and sum frequencies and tokens_N
    emlap_instances_grouped = (
        emlap_metadata_instances.groupby("Bidecade", dropna=True)[["Frequency", "tokens_N"]]
        .sum()
        .reset_index()
    )

    # Sort bidecade intervals numerically
    emlap_instances_grouped = emlap_instances_grouped.sort_values(
        by="Bidecade",
        key=lambda col: col.str.extract(r"(\d+)")[0].astype(int)
    )

    # Calculate relative frequency (Frequency / tokens_N)
    emlap_instances_grouped["RelativeFrequency"] = (
        emlap_instances_grouped["Frequency"] / emlap_instances_grouped["tokens_N"]
    )

    # Calculate relative frequency for individual works
    emlap_metadata_instances["RelativeFrequency"] = (
        emlap_metadata_instances["Frequency"] / emlap_metadata_instances["tokens_N"]
    )

    # Plot 1: Absolute frequency vs. bidecades
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(
        emlap_instances_grouped["Bidecade"],
        emlap_instances_grouped["Frequency"],
        color="blue"
    )
    ax1.set_xlabel("Bidecades", fontsize=12)
    ax1.set_ylabel("Absolute Frequency", fontsize=12)
    ax1.set_title(f"Absolute Frequency of '{entry}' by Bidecades", fontsize=14)
    ax1.set_xticks(range(len(emlap_instances_grouped["Bidecade"])))
    ax1.set_xticklabels(emlap_instances_grouped["Bidecade"], rotation=90)
    st.pyplot(fig1)

    # Plot 2: Relative frequency vs. bidecades
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.bar(
        emlap_instances_grouped["Bidecade"],
        emlap_instances_grouped["RelativeFrequency"],
        color="orange"
    )
    ax2.set_xlabel("Bidecades", fontsize=12)
    ax2.set_ylabel("Relative Frequency (Frequency / tokens_N)", fontsize=12)
    ax2.set_title(f"Relative Frequency of '{entry}' by Bidecades", fontsize=14)
    ax2.set_xticks(range(len(emlap_instances_grouped["Bidecade"])))
    ax2.set_xticklabels(emlap_instances_grouped["Bidecade"], rotation=90)
    st.pyplot(fig2)

    # Plot 3: Absolute frequency vs. works
    fig3, ax3 = plt.subplots(figsize=(15, 6))
    ax3.bar(
        emlap_metadata_instances["labeldate"],
        emlap_metadata_instances["Frequency"],
        color="green"
    )
    ax3.set_xlabel("Works", fontsize=12)
    ax3.set_ylabel("Absolute Frequency", fontsize=12)
    ax3.set_title(f"Absolute Frequency of '{entry}' by Works", fontsize=14)
    ax3.set_xticks(range(len(emlap_metadata_instances["labeldate"])))
    ax3.set_xticklabels(emlap_metadata_instances["labeldate"], rotation=90)
    st.pyplot(fig3)

    # Plot 4: Relative frequency vs. works
    fig4, ax4 = plt.subplots(figsize=(15, 6))
    ax4.bar(
        emlap_metadata_instances["labeldate"],
        emlap_metadata_instances["RelativeFrequency"],
        color="red"
    )
    ax4.set_xlabel("Works", fontsize=12)
    ax4.set_ylabel("Relative Frequency (Frequency / tokens_N)", fontsize=12)
    ax4.set_title(f"Relative Frequency of '{entry}' by Works", fontsize=14)
    ax4.set_xticks(range(len(emlap_metadata_instances["labeldate"])))
    ax4.set_xticklabels(emlap_metadata_instances["labeldate"], rotation=90)
    st.pyplot(fig4)

# Display the plots if the entry is valid
if st.button("Generate Plots"):
    make_plot_bar(entry)