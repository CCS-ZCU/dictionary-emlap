import requests
import re
import pandas as pd
import xml.etree.ElementTree as ET
import json
import os
import sys
import matplotlib.pyplot as plt
from collections import Counter
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
import duckdb
import unicodedata

from pathlib import Path
import re
import pandas as pd

conn = duckdb.connect('/srv/data/grela/grela_v0.6.duckdb', read_only=True)

# 128 GB RAM → give DuckDB plenty, but leave headroom for Python/OS/file cache
conn.execute("""
  SET memory_limit = '96GB';              -- or '80GB' if you run multiple jobs
  SET threads = 8;                        -- raise gradually (16/24/32) if stable
  SET preserve_insertion_order = false;
  PRAGMA temp_directory='/srv/data/duckdb_tmp';   -- fast SSD/NVMe
  PRAGMA max_temp_directory_size='2TB';          -- whatever your disk allows
""")


conn.execute("""
-- Full stream with a stable per-work order for the whole GreLa
CREATE OR REPLACE TEMP TABLE grela_full_stream AS
SELECT
  t.grela_id,
  t.sentence_id,
  s.position AS sentence_position,
  t.token_id,
  t.token_text,
  LOWER(t.lemma) AS lemma_lower,
  t.pos,
  t.ref,
  t.char_start,
  t.char_end,
  ROW_NUMBER() OVER (
    PARTITION BY t.grela_id
    ORDER BY s.position, t.char_start
  ) AS seq_full
FROM tokens t
JOIN works w  ON t.grela_id = w.grela_id
JOIN sentences s USING (sentence_id);
""")

conn.execute("""
-- Content-only stream aligned to full stream with precomputed next hops/grams
CREATE OR REPLACE TEMP TABLE grela_content_stream AS
WITH c AS (
  SELECT
    f.*,
    ROW_NUMBER() OVER (
      PARTITION BY f.grela_id
      ORDER BY f.sentence_position, f.char_start
    ) AS seq_content
  FROM grela_full_stream f
  WHERE f.lemma_lower IS NOT NULL
    AND f.pos <> 'PUNCT'
)
SELECT
  c.*,
  LEAD(c.lemma_lower, 1) OVER (PARTITION BY c.grela_id ORDER BY c.seq_content) AS l2,
  LEAD(c.lemma_lower, 2) OVER (PARTITION BY c.grela_id ORDER BY c.seq_content) AS l3,
  LEAD(c.seq_full,     1) OVER (PARTITION BY c.grela_id ORDER BY c.seq_content) AS next1_seq_full,
  LEAD(c.seq_full,     2) OVER (PARTITION BY c.grela_id ORDER BY c.seq_content) AS next2_seq_full,
  CASE WHEN LEAD(c.lemma_lower,1) OVER (PARTITION BY c.grela_id ORDER BY c.seq_content) IS NOT NULL
       THEN c.lemma_lower || ' ' || LEAD(c.lemma_lower,1) OVER (PARTITION BY c.grela_id ORDER BY c.seq_content)
  END AS n2,
  CASE WHEN LEAD(c.lemma_lower,2) OVER (PARTITION BY c.grela_id ORDER BY c.seq_content) IS NOT NULL
       THEN c.lemma_lower || ' ' || LEAD(c.lemma_lower,1) OVER (PARTITION BY c.grela_id ORDER BY c.seq_content) || ' ' ||
            LEAD(c.lemma_lower,2) OVER (PARTITION BY c.grela_id ORDER BY c.seq_content)
  END AS n3
FROM c;
""")

def concordance_for_target_across_sentences(
    conn,
    target_canonical: str | None,
    target_relemmatized: str | None,
    window: int = 10,
    include_tokens: bool = True,          # build/return kwic_tokens & target_sentence_tokens
    max_hits: Optional[int] = None,       # LIMIT for top rows (post-order)
    out_path: Optional[str] = None,       # stream directly to Parquet if set
    emlap_only: bool = True,              # NEW: if True → grela_id LIKE 'emlap%', else whole GreLa
):
    """
    Cross-sentence KWIC in GreLa with strict adjacency, searching BOTH lemma and token_text
    for BOTH target_canonical and target_relemmatized. Normalizes Latin: lowercase, strip
    diacritics, æ→ae, œ→oe, j→i, v→u, condenses whitespace.

    If emlap_only=True (default), restricts to works with grela_id LIKE 'emlap%'.
    If emlap_only=False, runs over the entire GreLa corpus.

    De-duplicates hits so each (grela_id, target_sentence_id, start_seq_full) appears once,
    preferring lemma>token and canonical>relemmatized.

    Returns (per row):
      - target_phrase         : list[str]  (token_texts of the matched span)
      - target_from           : 'canonical' | 'relemmatized'
      - matched_by            : 'lemma' | 'token'
      - target_lemmata        : list[str]
      - target_token_ids      : list[int]
      - target_kwic_idx       : list[int]  (0-based positions within kwic_tokens)
      - target_sent_idx       : list[int]  (0-based positions within target_sentence_tokens)
      - grela_id              : str
      - target_sentence_id    : str
      - start_sentence_id     : str
      - end_sentence_id       : str
      - kwic_text             : str
      - kwic_tokens           : list[struct] (only if include_tokens=True; else NULL)
      - target_sentence_text  : str
      - target_sentence_tokens: list[struct] (only if include_tokens=True; else NULL)

    If out_path is provided, writes Parquet via DuckDB COPY and returns None.
    Otherwise, returns a pandas DataFrame.
    """

    # ---------- Normalization helpers ----------
    def _strip_diacritics(s: str) -> str:
        return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

    def _latin_norm(s: str | None) -> str | None:
        if not s or not isinstance(s, str):
            return None
        s = s.strip().lower()
        s = _strip_diacritics(s)
        s = (s.replace("æ", "ae").replace("œ", "oe").replace("j", "i").replace("v", "u"))
        s = s.replace("_", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s or None

    def _prep(t: str | None):
        if not t:
            return 0, ("", "", ""), ""
        words = t.split()
        if not (1 <= len(words) <= 3):
            raise ValueError("Only 1–3-word targets supported (MAX_N=3).")
        w = tuple(words + ["", "", ""])[:3]
        return len(words), w, " ".join(words)

    tc = _latin_norm(target_canonical)
    tr = _latin_norm(target_relemmatized)
    if not tc and not tr:
        raise ValueError("Provide at least one of target_canonical or target_relemmatized.")

    tc_len, (tc_w1, tc_w2, tc_w3), tc_phrase = _prep(tc)
    tr_len, (tr_w1, tr_w2, tr_w3), tr_phrase = _prep(tr)

    # ---------- SQL-side normalizer ----------
    def NORM(expr: str) -> str:
        # lower + æ/œ/j/v normalization
        return ("replace(replace(replace(replace(lower({x}), 'æ', 'ae'), 'œ', 'oe'), 'j', 'i'), 'v', 'u')"
                .format(x=expr))

    # lemma→token fallback when lemma_lower is empty
    LEMMA_OR_TOKEN_NORM = NORM("coalesce(nullif(cs.lemma_lower, ''), cs.token_text)")

    # ---------- Corpus predicate ----------
    # If emlap_only: restrict to grela_id LIKE 'emlap%'
    # Else: use whole GreLa (no additional filter).
    corpus_pred = "grela_id LIKE 'emlap%'" if emlap_only else "TRUE"

    # ---------- SQL templates (heavy vs light) ----------
    # Common prelude through context + target_enrich (always needed)
    sql_core = f"""
WITH base_full AS (
  SELECT *
  FROM grela_full_stream
  WHERE {corpus_pred}
),
base_content AS (
  SELECT *
  FROM grela_content_stream
  WHERE {corpus_pred}
),
raw_matches AS (
  -- 1) lemma matches: canonical
  SELECT cs.grela_id, cs.sentence_id AS target_sentence_id, cs.seq_full AS start_seq_full,
         ?::INT AS target_len, 'lemma' AS matched_by, 'canonical' AS target_from, ?::VARCHAR AS target_phrase
  FROM base_content cs
  WHERE ? AND (
    (? = 1 AND {LEMMA_OR_TOKEN_NORM} = ?)
    OR (? = 2 AND {NORM('cs.n2')} = ?)
    OR (? = 3 AND {NORM('cs.n3')} = ?)
  )

  UNION ALL

  -- 2) lemma matches: relemmatized
  SELECT cs.grela_id, cs.sentence_id, cs.seq_full,
         ?::INT, 'lemma', 'relemmatized', ?::VARCHAR
  FROM base_content cs
  WHERE ? AND (
    (? = 1 AND {LEMMA_OR_TOKEN_NORM} = ?)
    OR (? = 2 AND {NORM('cs.n2')} = ?)
    OR (? = 3 AND {NORM('cs.n3')} = ?)
  )

  UNION ALL

  -- 3) token_text matches: canonical (strict adjacency)
  SELECT f1.grela_id, f1.sentence_id, f1.seq_full,
         ?::INT, 'token', 'canonical', ?::VARCHAR
  FROM base_full f1
  LEFT JOIN base_full f2
    ON f2.grela_id = f1.grela_id AND f2.seq_full = f1.seq_full + 1
  LEFT JOIN base_full f3
    ON f3.grela_id = f1.grela_id AND f3.seq_full = f1.seq_full + 2
  WHERE ? AND (
    (? = 1 AND {NORM('f1.token_text')} = ?)
    OR (? = 2 AND {NORM('f1.token_text')} = ? AND {NORM('f2.token_text')} = ?)
    OR (? = 3 AND {NORM('f1.token_text')} = ? AND {NORM('f2.token_text')} = ? AND {NORM('f3.token_text')} = ?)
  )

  UNION ALL

  -- 4) token_text matches: relemmatized (strict adjacency)
  SELECT f1.grela_id, f1.sentence_id, f1.seq_full,
         ?::INT, 'token', 'relemmatized', ?::VARCHAR
  FROM base_full f1
  LEFT JOIN base_full f2
    ON f2.grela_id = f1.grela_id AND f2.seq_full = f1.seq_full + 1
  LEFT JOIN base_full f3
    ON f3.grela_id = f1.grela_id AND f3.seq_full = f1.seq_full + 2
  WHERE ? AND (
    (? = 1 AND {NORM('f1.token_text')} = ?)
    OR (? = 2 AND {NORM('f1.token_text')} = ? AND {NORM('f2.token_text')} = ?)
    OR (? = 3 AND {NORM('f1.token_text')} = ? AND {NORM('f2.token_text')} = ? AND {NORM('f3.token_text')} = ?)
  )
),
ranked AS (
  SELECT *, ROW_NUMBER() OVER (
    PARTITION BY grela_id, target_sentence_id, start_seq_full
    ORDER BY
      CASE matched_by WHEN 'lemma' THEN 0 ELSE 1 END,
      CASE target_from WHEN 'canonical' THEN 0 ELSE 1 END
  ) AS rn
  FROM raw_matches
),
uniq_matches AS (
  SELECT grela_id, target_sentence_id, start_seq_full, target_len, matched_by, target_from, target_phrase
  FROM ranked
  WHERE rn = 1
),
bounds AS (
  SELECT m.grela_id, m.target_sentence_id, m.start_seq_full, m.target_len, m.matched_by, m.target_from, m.target_phrase,
         CASE m.target_len WHEN 1 THEN m.start_seq_full WHEN 2 THEN cs.next1_seq_full WHEN 3 THEN cs.next2_seq_full END AS end_seq_full
  FROM uniq_matches m
  JOIN base_content cs
    ON cs.grela_id = m.grela_id AND cs.seq_full = m.start_seq_full
  WHERE CASE m.target_len
          WHEN 1 THEN TRUE
          WHEN 2 THEN cs.next1_seq_full = m.start_seq_full + 1
          WHEN 3 THEN cs.next2_seq_full = m.start_seq_full + 2
        END
),
context AS (
  SELECT b.grela_id, b.target_sentence_id, b.start_seq_full, b.target_len, b.matched_by, b.target_from, b.target_phrase,
         f.sentence_id, f.token_id, f.token_text, f.lemma_lower, f.pos, f.ref, f.char_start, f.char_end,
         ROW_NUMBER() OVER (
           PARTITION BY b.grela_id, b.target_sentence_id, b.start_seq_full
           ORDER BY f.seq_full
         ) AS ord,
         (f.seq_full BETWEEN b.start_seq_full AND b.end_seq_full) AS is_target
  FROM bounds b
  JOIN base_full f
    ON f.grela_id = b.grela_id
   AND f.seq_full BETWEEN (b.start_seq_full - ?) AND (b.end_seq_full + ?)
),
needed_sentences AS (
  SELECT DISTINCT grela_id, target_sentence_id AS sentence_id
  FROM context
),
sentence_map AS (
  SELECT
    e.grela_id, e.sentence_id, e.token_id,
    ROW_NUMBER() OVER (PARTITION BY e.grela_id, e.sentence_id ORDER BY e.char_start) - 1 AS sent_idx0
  FROM base_full e
  JOIN needed_sentences n
    ON n.grela_id = e.grela_id AND n.sentence_id = e.sentence_id
),
target_enrich AS (
  SELECT
    c.grela_id, c.target_sentence_id, c.start_seq_full,
    LIST(c.token_text)   FILTER (WHERE c.is_target) AS target_phrase,
    LIST(c.lemma_lower)  FILTER (WHERE c.is_target) AS target_lemmata,
    LIST(c.token_id)     FILTER (WHERE c.is_target) AS target_token_ids,
    LIST(c.ord - 1)      FILTER (WHERE c.is_target) AS target_kwic_idx,
    LIST(sm.sent_idx0)   FILTER (WHERE c.is_target) AS target_sent_idx
  FROM context c
  LEFT JOIN sentence_map sm
    ON sm.grela_id = c.grela_id
   AND sm.sentence_id = c.sentence_id
   AND sm.token_id = c.token_id
  GROUP BY c.grela_id, c.target_sentence_id, c.start_seq_full
)
"""

    # Heavy variant (includes kwic_tokens + target_sentence_tokens)
    sql_heavy_tail = """
,agg_kwic AS (
  SELECT
    grela_id, target_sentence_id, start_seq_full,
    ANY_VALUE(target_len)   AS target_len,
    ANY_VALUE(matched_by)   AS matched_by,
    ANY_VALUE(target_from)  AS target_from,
    LIST(sentence_id ORDER BY ord)           AS window_sentence_ids,
    STRING_AGG(token_text, ' ' ORDER BY ord) AS kwic_text,
    LIST(
      STRUCT_PACK(
        token_id := token_id,
        token_text := token_text,
        lemma := lemma_lower,
        pos := pos,
        ref := ref,
        sentence_id := sentence_id,
        char_start := char_start,
        char_end := char_end
      )
      ORDER BY ord
    ) AS kwic_tokens
  FROM context
  GROUP BY grela_id, target_sentence_id, start_seq_full
),
target_sentence_texts AS (
  SELECT e.grela_id, e.sentence_id,
         STRING_AGG(e.token_text, ' ' ORDER BY e.char_start) AS sentence_text
  FROM base_full e
  JOIN needed_sentences n
    ON n.grela_id = e.grela_id AND n.sentence_id = e.sentence_id
  GROUP BY e.grela_id, e.sentence_id
),
target_sentence_tokens AS (
  SELECT e.grela_id, e.sentence_id,
         LIST(
           STRUCT_PACK(
             token_id := e.token_id,
             token_text := e.token_text,
             lemma := e.lemma_lower,
             pos := e.pos,
             ref := e.ref,
             char_start := e.char_start,
             char_end := e.char_end
           )
           ORDER BY e.char_start
         ) AS sentence_tokens
  FROM base_full e
  JOIN needed_sentences n
    ON n.grela_id = e.grela_id AND n.sentence_id = e.sentence_id
  GROUP BY e.grela_id, e.sentence_id
)
SELECT
  te.target_phrase,
  a.target_from,
  a.matched_by,
  te.target_lemmata,
  te.target_token_ids,
  te.target_kwic_idx,
  te.target_sent_idx,
  a.grela_id,
  a.target_sentence_id,
  a.window_sentence_ids[1]  AS start_sentence_id,
  a.window_sentence_ids[-1] AS end_sentence_id,
  a.kwic_text,
  a.kwic_tokens,
  tst.sentence_text         AS target_sentence_text,
  tstok.sentence_tokens     AS target_sentence_tokens
FROM agg_kwic a
JOIN target_enrich te
  ON te.grela_id = a.grela_id
 AND te.target_sentence_id = a.target_sentence_id
 AND te.start_seq_full = a.start_seq_full
LEFT JOIN target_sentence_texts  tst
  ON tst.grela_id = a.grela_id AND tst.sentence_id = a.target_sentence_id
LEFT JOIN target_sentence_tokens tstok
  ON tstok.grela_id = a.grela_id AND tstok.sentence_id = a.target_sentence_id
ORDER BY a.grela_id, a.target_sentence_id, a.start_seq_full
"""

    # Light variant (no kwic_tokens / target_sentence_tokens)
    sql_light_tail = """
,agg_kwic AS (
  SELECT
    grela_id, target_sentence_id, start_seq_full,
    ANY_VALUE(target_len)   AS target_len,
    ANY_VALUE(matched_by)   AS matched_by,
    ANY_VALUE(target_from)  AS target_from,
    LIST(sentence_id ORDER BY ord)           AS window_sentence_ids,
    STRING_AGG(token_text, ' ' ORDER BY ord) AS kwic_text
  FROM context
  GROUP BY grela_id, target_sentence_id, start_seq_full
),
target_sentence_texts AS (
  SELECT e.grela_id, e.sentence_id,
         STRING_AGG(e.token_text, ' ' ORDER BY e.char_start) AS sentence_text
  FROM base_full e
  JOIN needed_sentences n
    ON n.grela_id = e.grela_id AND n.sentence_id = e.sentence_id
  GROUP BY e.grela_id, e.sentence_id
)
SELECT
  te.target_phrase,
  a.target_from,
  a.matched_by,
  te.target_lemmata,
  te.target_token_ids,
  te.target_kwic_idx,
  te.target_sent_idx,
  a.grela_id,
  a.target_sentence_id,
  a.window_sentence_ids[1]  AS start_sentence_id,
  a.window_sentence_ids[-1] AS end_sentence_id,
  a.kwic_text,
  NULL                      AS kwic_tokens,
  tst.sentence_text         AS target_sentence_text,
  NULL                      AS target_sentence_tokens
FROM agg_kwic a
JOIN target_enrich te
  ON te.grela_id = a.grela_id
 AND te.target_sentence_id = a.target_sentence_id
 AND te.start_seq_full = a.start_seq_full
LEFT JOIN target_sentence_texts  tst
  ON tst.grela_id = a.grela_id AND tst.sentence_id = a.target_sentence_id
ORDER BY a.grela_id, a.target_sentence_id, a.start_seq_full
"""

    sql_tail = sql_heavy_tail if include_tokens else sql_light_tail
    sql = sql_core + sql_tail

    # Add LIMIT if requested
    if max_hits is not None:
        sql = sql + f"\nLIMIT {int(max_hits)}"

    # ---------- Bind parameters ----------
    params = [
        # 1) lemma canonical
        tc_len, tc_phrase or "", bool(tc),
        tc_len, (tc or ""), tc_len, (tc or ""), tc_len, (tc or ""),
        # 2) lemma relemmatized
        tr_len, tr_phrase or "", bool(tr),
        tr_len, (tr or ""), tr_len, (tr or ""), tr_len, (tr or ""),
        # 3) token canonical
        tc_len, tc_phrase or "", bool(tc),
        tc_len, tc_w1,
        tc_len, tc_w1, tc_w2,
        tc_len, tc_w1, tc_w2, tc_w3,
        # 4) token relemmatized
        tr_len, tr_phrase or "", bool(tr),
        tr_len, tr_w1,
        tr_len, tr_w1, tr_w2,
        tr_len, tr_w1, tr_w2, tr_w3,
        # window
        window, window,
    ]

    # ---------- Execute ----------
    if out_path:
        sql_nosemi = sql.rstrip().rstrip(';')
        out_quoted = "'" + out_path.replace("'", "''") + "'"
        conn.execute(f"COPY ({sql_nosemi}) TO {out_quoted} (FORMAT PARQUET);", params)
        return None
    else:
        return conn.execute(sql, params).fetch_df()