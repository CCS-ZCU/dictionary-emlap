# grela_helpers.py

import re
from typing import Optional
import duckdb
import os
import shutil
import uuid


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def make_connection(
    db_path: str = "/srv/data/grela/grela_v0.7.duckdb",
    read_only: bool = True,
) -> duckdb.DuckDBPyConnection:
    """
    Create and configure a DuckDB connection for GreLa.
    No temp tables are created here; everything works directly on `tokens`.
    """
    conn = duckdb.connect(db_path, read_only=read_only)
    conn.execute("""
      SET memory_limit = '96GB';
      SET threads = 8;
      SET preserve_insertion_order = false;
      PRAGMA temp_directory='/srv/data/duckdb_tmp';
      PRAGMA max_temp_directory_size='2TB';
    """)
    return conn


# ---------------------------------------------------------------------------
# Helpers for target normalization
# ---------------------------------------------------------------------------

def _norm(s: Optional[str]) -> Optional[str]:
    """
    Light normalization: lowercase + collapse whitespace.
    (No diacritics / j→i / v→u; your DB is already normalized.)
    """
    if not s or not isinstance(s, str):
        return None
    s = re.sub(r"\s+", " ", s.strip().lower())
    return s or None


def _prep(t: Optional[str]):
    """
    Prepare 1–3 word target:
      returns (length, (w1,w2,w3), "joined phrase")
    Raises if >3 words.
    """
    if not t:
        return 0, ("", "", ""), ""
    words = t.split()
    if not (1 <= len(words) <= 3):
        raise ValueError("Only 1–3-word targets supported (MAX_N=3).")
    w = tuple(words + ["", "", ""])[:3]
    return len(words), w, " ".join(words)


# ---------------------------------------------------------------------------
# Phase 1: find token spans (only IDs)
# ---------------------------------------------------------------------------

def create_target_spans_table(
    conn: duckdb.DuckDBPyConnection,
    table_name: str = "temp_target_spans",
    target_canonical: Optional[str] = None,
    target_relemmatized: Optional[str] = None,
    emlap_only: bool = True,
    max_hits: Optional[int] = None,
) -> None:
    """
    Phase 1: find target spans ONLY and materialize them as a TEMP TABLE in DuckDB.

    The resulting table has schema:

      table_name(
        span_idx           BIGINT,  -- 1,2,3,... for chunked processing
        grela_id           TEXT,
        target_sentence_id TEXT,
        start_token_id     BIGINT,
        end_token_id       BIGINT,
        matched_by         TEXT,    -- 'lemma' | 'token'
        target_from        TEXT,    -- 'canonical' | 'relemmatized'
        target_phrase      TEXT
      )

    Matching behaviour:
      - lemma_or_token := lower(COALESCE(NULLIF(lemma,''), token_text))
      - token_text_lower := lower(token_text)
      - 1–3-word collocations, strict adjacency via token_id+1/+2.
      - de-duplicates on (grela_id, sentence_id, start_token_id), preferring
        lemma > token, canonical > relemmatized (same as your previous logic).
    """

    tc = _norm(target_canonical)
    tr = _norm(target_relemmatized)
    if not tc and not tr:
        raise ValueError("Provide at least one of target_canonical or target_relemmatized.")

    tc_len, (tc_w1, tc_w2, tc_w3), tc_phrase = _prep(tc)
    tr_len, (tr_w1, tr_w2, tr_w3), tr_phrase = _prep(tr)

    corpus_pred = "w.grela_id LIKE 'emlap%'" if emlap_only else "TRUE"

    raw_selects: list[str] = []
    params: list = []

    def add_select(sql_part: str, *p):
        raw_selects.append(sql_part)
        params.extend(p)

    # ---- 1) lemma-or-token canonical ----
    if tc:
        if tc_len == 1:
            add_select(
                f"""
    SELECT
      t1.grela_id,
      t1.sentence_id AS target_sentence_id,
      t1.token_id    AS start_token_id,
      t1.token_id    AS end_token_id,
      'lemma'        AS matched_by,
      'canonical'    AS target_from,
      ?::VARCHAR     AS target_phrase
    FROM tokens t1
    JOIN works w ON w.grela_id = t1.grela_id
    WHERE {corpus_pred}
      AND lower(coalesce(nullif(t1.lemma, ''), t1.token_text)) = ?
                """,
                tc_phrase, tc_w1,
            )
        elif tc_len == 2:
            add_select(
                f"""
    SELECT
      t1.grela_id,
      t1.sentence_id AS target_sentence_id,
      t1.token_id    AS start_token_id,
      t2.token_id    AS end_token_id,
      'lemma'        AS matched_by,
      'canonical'    AS target_from,
      ?::VARCHAR     AS target_phrase
    FROM tokens t1
    JOIN tokens t2
      ON t2.grela_id = t1.grela_id
     AND t2.token_id = t1.token_id + 1
    JOIN works w ON w.grela_id = t1.grela_id
    WHERE {corpus_pred}
      AND lower(coalesce(nullif(t1.lemma, ''), t1.token_text)) = ?
      AND lower(coalesce(nullif(t2.lemma, ''), t2.token_text)) = ?
                """,
                tc_phrase, tc_w1, tc_w2,
            )
        elif tc_len == 3:
            add_select(
                f"""
    SELECT
      t1.grela_id,
      t1.sentence_id AS target_sentence_id,
      t1.token_id    AS start_token_id,
      t3.token_id    AS end_token_id,
      'lemma'        AS matched_by,
      'canonical'    AS target_from,
      ?::VARCHAR     AS target_phrase
    FROM tokens t1
    JOIN tokens t2
      ON t2.grela_id = t1.grela_id
     AND t2.token_id = t1.token_id + 1
    JOIN tokens t3
      ON t3.grela_id = t1.grela_id
     AND t3.token_id = t1.token_id + 2
    JOIN works w ON w.grela_id = t1.grela_id
    WHERE {corpus_pred}
      AND lower(coalesce(nullif(t1.lemma, ''), t1.token_text)) = ?
      AND lower(coalesce(nullif(t2.lemma, ''), t2.token_text)) = ?
      AND lower(coalesce(nullif(t3.lemma, ''), t3.token_text)) = ?
                """,
                tc_phrase, tc_w1, tc_w2, tc_w3,
            )

    # ---- 2) lemma-or-token relemmatized ----
    if tr:
        if tr_len == 1:
            add_select(
                f"""
    SELECT
      t1.grela_id,
      t1.sentence_id AS target_sentence_id,
      t1.token_id    AS start_token_id,
      t1.token_id    AS end_token_id,
      'lemma'        AS matched_by,
      'relemmatized' AS target_from,
      ?::VARCHAR     AS target_phrase
    FROM tokens t1
    JOIN works w ON w.grela_id = t1.grela_id
    WHERE {corpus_pred}
      AND lower(coalesce(nullif(t1.lemma, ''), t1.token_text)) = ?
                """,
                tr_phrase, tr_w1,
            )
        elif tr_len == 2:
            add_select(
                f"""
    SELECT
      t1.grela_id,
      t1.sentence_id AS target_sentence_id,
      t1.token_id    AS start_token_id,
      t2.token_id    AS end_token_id,
      'lemma'        AS matched_by,
      'relemmatized' AS target_from,
      ?::VARCHAR     AS target_phrase
    FROM tokens t1
    JOIN tokens t2
      ON t2.grela_id = t1.grela_id
     AND t2.token_id = t1.token_id + 1
    JOIN works w ON w.grela_id = t1.grela_id
    WHERE {corpus_pred}
      AND lower(coalesce(nullif(t1.lemma, ''), t1.token_text)) = ?
      AND lower(coalesce(nullif(t2.lemma, ''), t2.token_text)) = ?
                """,
                tr_phrase, tr_w1, tr_w2,
            )
        elif tr_len == 3:
            add_select(
                f"""
    SELECT
      t1.grela_id,
      t1.sentence_id AS target_sentence_id,
      t1.token_id    AS start_token_id,
      t3.token_id    AS end_token_id,
      'lemma'        AS matched_by,
      'relemmatized' AS target_from,
      ?::VARCHAR     AS target_phrase
    FROM tokens t1
    JOIN tokens t2
      ON t2.grela_id = t1.grela_id
     AND t2.token_id = t1.token_id + 1
    JOIN tokens t3
      ON t3.grela_id = t1.grela_id
     AND t3.token_id = t1.token_id + 2
    JOIN works w ON w.grela_id = t1.grela_id
    WHERE {corpus_pred}
      AND lower(coalesce(nullif(t1.lemma, ''), t1.token_text)) = ?
      AND lower(coalesce(nullif(t2.lemma, ''), t2.token_text)) = ?
      AND lower(coalesce(nullif(t3.lemma, ''), t3.token_text)) = ?
                """,
                tr_phrase, tr_w1, tr_w2, tr_w3,
            )

    # ---- 3) token_text canonical ----
    if tc:
        if tc_len == 1:
            add_select(
                f"""
    SELECT
      t1.grela_id,
      t1.sentence_id AS target_sentence_id,
      t1.token_id    AS start_token_id,
      t1.token_id    AS end_token_id,
      'token'        AS matched_by,
      'canonical'    AS target_from,
      ?::VARCHAR     AS target_phrase
    FROM tokens t1
    JOIN works w ON w.grela_id = t1.grela_id
    WHERE {corpus_pred}
      AND lower(t1.token_text) = ?
                """,
                tc_phrase, tc_w1,
            )
        elif tc_len == 2:
            add_select(
                f"""
    SELECT
      t1.grela_id,
      t1.sentence_id AS target_sentence_id,
      t1.token_id    AS start_token_id,
      t2.token_id    AS end_token_id,
      'token'        AS matched_by,
      'canonical'    AS target_from,
      ?::VARCHAR     AS target_phrase
    FROM tokens t1
    JOIN tokens t2
      ON t2.grela_id = t1.grela_id
     AND t2.token_id = t1.token_id + 1
    JOIN works w ON w.grela_id = t1.grela_id
    WHERE {corpus_pred}
      AND lower(t1.token_text) = ?
      AND lower(t2.token_text) = ?
                """,
                tc_phrase, tc_w1, tc_w2,
            )
        elif tc_len == 3:
            add_select(
                f"""
    SELECT
      t1.grela_id,
      t1.sentence_id AS target_sentence_id,
      t1.token_id    AS start_token_id,
      t3.token_id    AS end_token_id,
      'token'        AS matched_by,
      'canonical'    AS target_from,
      ?::VARCHAR     AS target_phrase
    FROM tokens t1
    JOIN tokens t2
      ON t2.grela_id = t1.grela_id
     AND t2.token_id = t1.token_id + 1
    JOIN tokens t3
      ON t3.grela_id = t1.grela_id
     AND t3.token_id = t1.token_id + 2
    JOIN works w ON w.grela_id = t1.grela_id
    WHERE {corpus_pred}
      AND lower(t1.token_text) = ?
      AND lower(t2.token_text) = ?
      AND lower(t3.token_text) = ?
                """,
                tc_phrase, tc_w1, tc_w2, tc_w3,
            )

    # ---- 4) token_text relemmatized ----
    if tr:
        if tr_len == 1:
            add_select(
                f"""
    SELECT
      t1.grela_id,
      t1.sentence_id AS target_sentence_id,
      t1.token_id    AS start_token_id,
      t1.token_id    AS end_token_id,
      'token'        AS matched_by,
      'relemmatized' AS target_from,
      ?::VARCHAR     AS target_phrase
    FROM tokens t1
    JOIN works w ON w.grela_id = t1.grela_id
    WHERE {corpus_pred}
      AND lower(t1.token_text) = ?
                """,
                tr_phrase, tr_w1,
            )
        elif tr_len == 2:
            add_select(
                f"""
    SELECT
      t1.grela_id,
      t1.sentence_id AS target_sentence_id,
      t1.token_id    AS start_token_id,
      t2.token_id    AS end_token_id,
      'token'        AS matched_by,
      'relemmatized' AS target_from,
      ?::VARCHAR     AS target_phrase
    FROM tokens t1
    JOIN tokens t2
      ON t2.grela_id = t1.grela_id
     AND t2.token_id = t1.token_id + 1
    JOIN works w ON w.grela_id = t1.grela_id
    WHERE {corpus_pred}
      AND lower(t1.token_text) = ?
      AND lower(t2.token_text) = ?
                """,
                tr_phrase, tr_w1, tr_w2,
            )
        elif tr_len == 3:
            add_select(
                f"""
    SELECT
      t1.grela_id,
      t1.sentence_id AS target_sentence_id,
      t1.token_id    AS start_token_id,
      t3.token_id    AS end_token_id,
      'token'        AS matched_by,
      'relemmatized' AS target_from,
      ?::VARCHAR     AS target_phrase
    FROM tokens t1
    JOIN tokens t2
      ON t2.grela_id = t1.grela_id
     AND t2.token_id = t1.token_id + 1
    JOIN tokens t3
      ON t3.grela_id = t1.grela_id
     AND t3.token_id = t1.token_id + 2
    JOIN works w ON w.grela_id = t1.grela_id
    WHERE {corpus_pred}
      AND lower(t1.token_text) = ?
      AND lower(t2.token_text) = ?
      AND lower(t3.token_text) = ?
                """,
                tr_phrase, tr_w1, tr_w2, tr_w3,
            )

    if not raw_selects:
        raise RuntimeError("No match branches active – this should not happen.")

    raw_matches_sql = "\nUNION ALL\n".join(raw_selects)
    limit_clause = f"\nLIMIT {int(max_hits)}" if max_hits is not None else ""

    sql = f"""
CREATE OR REPLACE TEMP TABLE {table_name} AS
WITH raw_matches AS (
{raw_matches_sql}
),
ranked AS (
  SELECT *,
         ROW_NUMBER() OVER (
           PARTITION BY grela_id, target_sentence_id, start_token_id
           ORDER BY
             CASE matched_by WHEN 'lemma' THEN 0 ELSE 1 END,
             CASE target_from WHEN 'canonical' THEN 0 ELSE 1 END
         ) AS rn
  FROM raw_matches
),
uniq_matches AS (
  SELECT
    grela_id,
    target_sentence_id,
    start_token_id,
    end_token_id,
    matched_by,
    target_from,
    target_phrase
  FROM ranked
  WHERE rn = 1
),
spans AS (
  SELECT
    ROW_NUMBER() OVER (
      ORDER BY grela_id, target_sentence_id, start_token_id
    ) AS span_idx,
    *
  FROM uniq_matches
)
SELECT *
FROM spans
{limit_clause};
"""

    conn.execute(sql, params)


# ---------------------------------------------------------------------------
# Phase 2: extract concordance (KWIC) from spans
# ---------------------------------------------------------------------------

def kwic_from_spans(
    conn: duckdb.DuckDBPyConnection,
    spans_table: str = "temp_target_spans",
    window: int = 10,
    include_tokens: bool = True,
    span_idx_min: Optional[int] = None,
    span_idx_max: Optional[int] = None,
    max_hits: Optional[int] = None,
    out_path: Optional[str] = None,
    # NEW (safe defaults; tune as needed):
    batch_threshold: int = 100_000,
    batch_size: int = 10_000,
    tmp_dir: Optional[str] = None,
):
    """
    Phase 2: Take a spans table (created by create_target_spans_table) and
    build full KWIC.

    If out_path is provided and the spans table is large, the function will
    automatically process spans in batches and merge them into a single Parquet
    (no global ordering, same schema).
    """

    def _q(path: str) -> str:
        # SQL-safe single-quoted path
        return "'" + path.replace("'", "''") + "'"

    # ------------------------------------------------------------------
    # Batched mode (only when writing to Parquet, and only in "full" mode)
    # ------------------------------------------------------------------
    if out_path and (span_idx_min is None and span_idx_max is None):
        n_spans = conn.execute(f"SELECT COUNT(*) FROM {spans_table}").fetchone()[0] or 0
        if max_hits is not None:
            n_spans = min(n_spans, int(max_hits))

        if n_spans > batch_threshold:
            out_path = str(out_path)

            # Choose a temp directory near output by default (usually faster / same FS)
            if tmp_dir is None:
                base = os.path.dirname(out_path) or "."
                tmp_dir = os.path.join(base, f".tmp_kwic_{uuid.uuid4().hex}")
            os.makedirs(tmp_dir, exist_ok=True)

            try:
                # Write per-batch Parquets
                for lo in range(1, n_spans + 1, batch_size):
                    hi = min(lo + batch_size - 1, n_spans)
                    part_path = os.path.join(tmp_dir, f"part_{lo:09d}_{hi:09d}.parquet")

                    # recurse in range mode (won't re-batch)
                    kwic_from_spans(
                        conn,
                        spans_table=spans_table,
                        window=window,
                        include_tokens=include_tokens,
                        span_idx_min=lo,
                        span_idx_max=hi,
                        max_hits=None,  # range wins
                        out_path=part_path,
                        batch_threshold=batch_threshold,
                        batch_size=batch_size,
                        tmp_dir=tmp_dir,
                    )

                # Merge parts -> final parquet INSIDE DuckDB, without ORDER BY (memory-friendly)
                glob = os.path.join(tmp_dir, "part_*.parquet")
                conn.execute(
                    f"""
                    COPY (
                      SELECT *
                      FROM read_parquet({_q(glob)})
                    ) TO {_q(out_path)} (FORMAT PARQUET);
                    """
                )
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            return None

    # ------------------------------------------------------------------
    # Non-batched execution (single chunk OR explicit span range)
    # ------------------------------------------------------------------
    span_filter = ""
    span_params: list = []

    if span_idx_min is not None and span_idx_max is not None:
        span_filter = "WHERE span_idx BETWEEN ? AND ?"
        span_params = [span_idx_min, span_idx_max]
        limit_clause = ""  # range wins; no extra LIMIT
    else:
        span_filter = ""
        span_params = []
        limit_clause = f"LIMIT {int(max_hits)}" if max_hits is not None else ""

    # NOTE: removed ORDER BY in uniq_matches to avoid global sort work
    sql_core = f"""
WITH uniq_matches AS (
  SELECT *
  FROM {spans_table}
  {span_filter}
  {limit_clause}
),
context AS (
  SELECT
    m.grela_id,
    m.target_sentence_id,
    m.start_token_id,
    m.end_token_id,
    m.matched_by,
    m.target_from,
    m.target_phrase,
    t.sentence_id,
    t.token_id,
    t.token_text,
    LOWER(t.lemma) AS lemma_lower,
    t.pos,
    t.ref,
    t.char_start,
    t.char_end,
    ROW_NUMBER() OVER (
      PARTITION BY m.grela_id, m.target_sentence_id, m.start_token_id
      ORDER BY t.token_id
    ) AS ord,
    (t.token_id BETWEEN m.start_token_id AND m.end_token_id) AS is_target
  FROM uniq_matches m
  JOIN tokens t
    ON t.grela_id = m.grela_id
   AND t.token_id BETWEEN (m.start_token_id - ?) AND (m.end_token_id + ?)
),
needed_sentences AS (
  SELECT DISTINCT grela_id, target_sentence_id AS sentence_id
  FROM context
),
sentence_map AS (
  SELECT
    t.grela_id,
    t.sentence_id,
    t.token_id,
    ROW_NUMBER() OVER (
      PARTITION BY t.grela_id, t.sentence_id
      ORDER BY t.char_start
    ) - 1 AS sent_idx0
  FROM tokens t
  JOIN needed_sentences n
    ON n.grela_id = t.grela_id AND n.sentence_id = t.sentence_id
),
target_enrich AS (
  SELECT
    c.grela_id,
    c.target_sentence_id,
    c.start_token_id,
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
  GROUP BY c.grela_id, c.target_sentence_id, c.start_token_id
)
"""

    # ---------- heavy tail (with kwic_tokens & sentence_tokens) ----------
    # NOTE: removed final ORDER BY in the SELECT to avoid global sorting
    sql_heavy_tail = """
,agg_kwic AS (
  SELECT
    grela_id, target_sentence_id, start_token_id,
    ANY_VALUE(matched_by)    AS matched_by,
    ANY_VALUE(target_from)   AS target_from,
    ANY_VALUE(target_phrase) AS target_phrase,
    LIST(sentence_id ORDER BY ord)            AS window_sentence_ids,
    STRING_AGG(token_text, ' ' ORDER BY ord)  AS kwic_text,
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
  GROUP BY grela_id, target_sentence_id, start_token_id
),
target_sentence_texts AS (
  SELECT
    t.grela_id,
    t.sentence_id,
    STRING_AGG(t.token_text, ' ' ORDER BY t.char_start) AS sentence_text
  FROM tokens t
  JOIN needed_sentences n
    ON n.grela_id = t.grela_id AND n.sentence_id = t.sentence_id
  GROUP BY t.grela_id, t.sentence_id
),
target_sentence_tokens AS (
  SELECT
    t.grela_id,
    t.sentence_id,
    LIST(
      STRUCT_PACK(
        token_id := t.token_id,
        token_text := t.token_text,
        lemma := LOWER(t.lemma),
        pos := t.pos,
        ref := t.ref,
        char_start := t.char_start,
        char_end := t.char_end
      )
      ORDER BY t.char_start
    ) AS sentence_tokens
  FROM tokens t
  JOIN needed_sentences n
    ON n.grela_id = t.grela_id AND n.sentence_id = t.sentence_id
  GROUP BY t.grela_id, t.sentence_id
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
  tstok.sentence_tokens     AS target_sentence_tokens,
  w.author,
  w.title,
  w.not_before,
  w.not_after
FROM agg_kwic a
JOIN target_enrich te
  ON te.grela_id = a.grela_id
 AND te.target_sentence_id = a.target_sentence_id
 AND te.start_token_id = a.start_token_id
LEFT JOIN target_sentence_texts  tst
  ON tst.grela_id = a.grela_id AND tst.sentence_id = a.target_sentence_id
LEFT JOIN target_sentence_tokens tstok
  ON tstok.grela_id = a.grela_id AND tstok.sentence_id = a.target_sentence_id
LEFT JOIN works w
  ON w.grela_id = a.grela_id
"""

    # ---------- light tail (no kwic_tokens / sentence_tokens) ----------
    sql_light_tail = """
,agg_kwic AS (
  SELECT
    grela_id, target_sentence_id, start_token_id,
    ANY_VALUE(matched_by)    AS matched_by,
    ANY_VALUE(target_from)   AS target_from,
    ANY_VALUE(target_phrase) AS target_phrase,
    LIST(sentence_id ORDER BY ord)            AS window_sentence_ids,
    STRING_AGG(token_text, ' ' ORDER BY ord)  AS kwic_text
  FROM context
  GROUP BY grela_id, target_sentence_id, start_token_id
),
target_sentence_texts AS (
  SELECT
    t.grela_id,
    t.sentence_id,
    STRING_AGG(t.token_text, ' ' ORDER BY t.char_start) AS sentence_text
  FROM tokens t
  JOIN needed_sentences n
    ON n.grela_id = t.grela_id AND n.sentence_id = t.sentence_id
  GROUP BY t.grela_id, t.sentence_id
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
  NULL                      AS target_sentence_tokens,
  w.author,
  w.title,
  w.not_before,
  w.not_after
FROM agg_kwic a
JOIN target_enrich te
  ON te.grela_id = a.grela_id
 AND te.target_sentence_id = a.target_sentence_id
 AND te.start_token_id = a.start_token_id
LEFT JOIN target_sentence_texts  tst
  ON tst.grela_id = a.grela_id AND tst.sentence_id = a.target_sentence_id
LEFT JOIN works w
  ON w.grela_id = a.grela_id
"""

    sql_tail = sql_heavy_tail if include_tokens else sql_light_tail
    sql = sql_core + sql_tail

    params = span_params + [window, window]

    if out_path:
        sql_nosemi = sql.rstrip().rstrip(";")
        path_str = str(out_path)
        conn.execute(f"COPY ({sql_nosemi}) TO {_q(path_str)} (FORMAT PARQUET);", params)
        return None
    else:
        return conn.execute(sql, params).fetch_df()