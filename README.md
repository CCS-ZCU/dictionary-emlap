#  Ruland in Usage

---
## Authors
* Vojtěch Kaše [![](https://orcid.org/sites/default/files/images/orcid_16x16.png)](https://orcid.org/0000-0002-6601-1605)
* Sarah Lang  [![](https://orcid.org/sites/default/files/images/orcid_16x16.png)](https://orcid.org/0000-0002-4618-9481)

## License
CC-BY-SA 4.0, see attached License.md

---
## Description

This repository serves as an supplementary to the article "Contextual Word Embeddings for Paracelsian Lexicography: Tangled Terminologies and their Origins in Ruland’s Alchemical Dictionary". 

It hosts code, data and figures in which we:
(1) load, parse, and lemmatize entries from a TEI-XML edition of Ruland's dictionary available from [here](https://github.com/sarahalang/alchemical-dictionaries/tree/main)
(2) detect all these entries in the Latin portition of [Grela](https://zenodo.org/records/18160596), with a special attention paid to instances found in [EMLAP](https://zenodo.org/records/14765511), a corpus of Early Modern Latin Alchemical Prints also included in GreLa
(3) load these data into a [web app](https://ccs-lab.zcu.cz/ruland-plots/), which allows to visualize temporal distribution and token embeddings of the detected instances
(4) conduct additional analyses of the data.

## Getting started

```bash
git clone [url-of-the-git-file]
cd [name-of-the-repo]
pip install -r requirements.txt
```

Go to `scripts` and run:
```bash
streamlit run  ruland-plots_streamlit.py --server.address localhost --server.port 8065 --browser.gatherUsageStats False
```
## How to cite

Please refer to the Zenodo release of this repo and to the related article.

