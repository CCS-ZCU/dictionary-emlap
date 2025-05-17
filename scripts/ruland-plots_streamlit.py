# Streamlit App for Visualizing Ruland's Entries
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import streamlit as st
import fnmatch

# Title and description for the app
st.title("Ruland in Usage")
st.markdown(
    """
    This app visualizes the distribution of instances from Ruland's entries by bidecades aggregates and across individual works from the EMLAP corpus. It gives you both absolute and relative values.

    The source code for the app is available from [here](https://github.com/CCS-ZCU/dictionary-emlap).

    Enter an entry (e.g., **Mercurius metallorum**) to see the plots.
    """
)


# Load data files with caching
@st.cache_data
def load_data():
    # Load entries JSON file
    entries_df = pd.read_json("../data/large_files/ruland-emlap.json")

    # Remove duplicate rows based on 'Lemma' and 'emlap_instances_N' (or all columns if preferred)
    entries_df = entries_df.drop_duplicates(subset=["Lemma", "emlap_instances_N"])

    # Load metadata from the web
    emlap_metadata = pd.read_csv(
        "https://raw.githubusercontent.com/CCS-ZCU/EMLAP_ETL/refs/heads/master/data/emlap_metadata.csv",
        sep=";"
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


# Function to find matching keys with corresponding 'emlap_instances_N' values
def find_matching_keys(pattern, entries_df):
    """
    Finds matching keys in entries_df['Lemma'] with their corresponding
    'emlap_instances_N' values, sorted from highest to lowest.

    Args:
        pattern (str): The search pattern (supports wildcards '*').
        entries_df (DataFrame): DataFrame containing 'Lemma' and 'emlap_instances_N' columns.

    Returns:
        list: List of tuples in the format [(key, n), ...] where 'key' is the lemma
              and 'n' is the corresponding emlap_instances_N value, sorted by 'n' descending.
    """
    # Validate input: Ensure entries_df contains the required columns
    if not all(col in entries_df.columns for col in ["Lemma", "emlap_instances_N"]):
        raise ValueError("entries_df must contain 'Lemma' and 'emlap_instances_N' columns.")

    # Filter DataFrame for valid rows (clean NaNs)
    filtered_df = entries_df.dropna(subset=["Lemma", "emlap_instances_N"]).copy()

    # Ensure column types are consistent
    filtered_df["Lemma"] = filtered_df["Lemma"].astype(str)
    filtered_df["emlap_instances_N"] = filtered_df["emlap_instances_N"].astype(int)

    # Match keys using wildcard (*) or substring
    if "*" in pattern:
        filtered_df = filtered_df[
            filtered_df["Lemma"].str.contains(fnmatch.translate(pattern), case=False, regex=True)
        ]
    else:
        filtered_df = filtered_df[
            filtered_df["Lemma"].str.contains(pattern, case=False, na=False)
        ]

    # Create the result as sorted tuple list [(key, n), ...]
    result = filtered_df.sort_values("emlap_instances_N", ascending=False).apply(
        lambda row: (row["Lemma"], row["emlap_instances_N"]), axis=1
    ).tolist()

    # Return the sorted list of matches
    return result


# Function to generate plots for a selected entry
def generate_plots(selected_entry):
    # Get instances IDs for the selected entry
    instances = entries_df[entries_df["Lemma"] == selected_entry]["instances_ids"]
    instances_texts_data = entries_df[entries_df["Lemma"] == selected_entry]["emlap_instances"]

    # Handle cases where there is no data for the selected entry
    if len(instances) == 0 or len(instances_texts_data) == 0:
        st.error("No data available for the selected entry.")
        return

    # Convert instances and texts into usable formats
    instances_ids = instances.tolist()[0]  # Get the list of instance IDs
    instances_texts_list = instances_texts_data.tolist()[0]  # Get the list of instance details

    # Create a DataFrame from the instances texts list
    instances_df = pd.DataFrame(
        instances_texts_list,
        columns=["emlap ID", "sentence index", "sentence text"]
    )

    # Count the frequency of each instance ID
    counter = Counter(instances_ids)

    # Copy metadata and calculate frequency
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

    # Absolute frequency by bidecades
    st.subheader(f"Absolute Frequency of '{selected_entry}' by Bidecades")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(
        emlap_instances_grouped["Bidecade"],
        emlap_instances_grouped["Frequency"],
        color="blue"
    )
    ax1.set_xlabel("Bidecades", fontsize=12)
    ax1.set_ylabel("Absolute Frequency", fontsize=12)
    ax1.set_xticklabels(emlap_instances_grouped["Bidecade"], rotation=90)
    st.pyplot(fig1)

    # Relative frequency by bidecades
    st.subheader(f"Relative Frequency of '{selected_entry}' by Bidecades")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.bar(
        emlap_instances_grouped["Bidecade"],
        emlap_instances_grouped["RelativeFrequency"],
        color="orange"
    )
    ax2.set_xlabel("Bidecades", fontsize=12)
    ax2.set_ylabel("Relative Frequency (Frequency / tokens_N)", fontsize=12)
    ax2.set_xticklabels(emlap_instances_grouped["Bidecade"], rotation=90)
    st.pyplot(fig2)

    # Absolute frequency by works
    st.subheader(f"Absolute Frequency of '{selected_entry}' by Works")
    fig3, ax3 = plt.subplots(figsize=(15, 6))
    ax3.bar(
        emlap_metadata_instances["labeldate"],
        emlap_metadata_instances["Frequency"],
        color="green"
    )
    ax3.set_xlabel("Works", fontsize=12)
    ax3.set_ylabel("Absolute Frequency", fontsize=12)
    ax3.set_xticklabels(emlap_metadata_instances["labeldate"], rotation=90)
    st.pyplot(fig3)

    # Relative frequency by works
    st.subheader(f"Relative Frequency of '{selected_entry}' by Works")
    fig4, ax4 = plt.subplots(figsize=(15, 6))
    ax4.bar(
        emlap_metadata_instances["labeldate"],
        emlap_metadata_instances["Frequency"] / emlap_metadata_instances["tokens_N"],
        color="red"
    )
    ax4.set_xlabel("Works", fontsize=12)
    ax4.set_ylabel("Relative Frequency (Frequency / tokens_N)", fontsize=12)
    ax4.set_xticklabels(emlap_metadata_instances["labeldate"], rotation=90)
    st.pyplot(fig4)

    # Display the instances DataFrame
    st.subheader(f"Detailed Sentences for '{selected_entry}'")
    st.write(instances_df)

    # Optional: Add a download button for the instances DataFrame
    csv_download = instances_df.to_csv(index=False)
    st.download_button(
        label=f"Download Sentences for '{selected_entry}'",
        data=csv_download,
        file_name=f"{selected_entry}_sentences.csv",
        mime="text/csv"
    )

# User input
entry = st.text_input("Ruland's entry", value="Mercurius metallorum")
matching_keys = find_matching_keys(entry, entries_df)

if len(matching_keys) == 0:
    st.error("No entries found. Try a different search.")
elif len(matching_keys) == 1:
    st.success(f"One match found: {matching_keys[0][0]}")
    generate_plots(matching_keys[0][0])
else:
    st.warning("Multiple matches found. Select one below:")
    selected_entry = st.selectbox(
        "Select an entry:", [f"{key} ({count})" for key, count in matching_keys]
    )
    if st.button("Generate Plots"):
        generate_plots(selected_entry.split("(")[0].strip())