#  Ruland in Usage

---
## Authors
* Sarah A. Lang
* Vojtěch Kaše
* Georgiana Hedesan
* Petr Pavlas


## License
CC-BY-SA 4.0, see attached License.md

---
## Description

This repository hosts code in which we:
(1) load, parse, and lemmatize entries from a TEI-XML edition of Ruland's dictionary available from [here](https://github.com/sarahalang/alchemical-dictionaries/tree/main)
(2) detect all these entries in the [EMLAP](https://zenodo.org/records/14765511) corpus of Early Modern Latin Alchemical Prints
(3) load these data into a web app allowing to visualize temporal distribution of the detected instances
(4) conduct additional analyses of the data

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

[once a release is created and published via zenodo, put its citation here]

## Ackwnowledgement

[This work has been supported by ...]
