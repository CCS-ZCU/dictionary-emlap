

def process_kwic_tokens(
    kwic_tokens: list[dict],
    target_kwic_idx: list[int],
    *,
    context_lemmatized: bool = True,     # if False -> use token_text
    target_lemmatized: bool = False,     # if True  -> use lemma for the target phrase
    context_pos_filtered: bool = True,   # if True  -> keep only POS in pos_keep for context
    pos_keep: tuple[str, ...] = ("NOUN", "ADJ", "VERB", "PROPN"),
):
    """
    Split a KWIC window into LEFT CONTEXT, TARGET PHRASE, RIGHT CONTEXT, with optional
    lemmatization and POS filtering for the context, and optional lemmatization for the target.

    Inputs
    ------
    kwic_tokens: list[dict]
        Each dict must have at least: 'token_text', 'lemma', 'pos'
    target_kwic_idx: list[int]
        KWIC indices of the target tokens (usually contiguous, but not required)

    Returns
    -------
    out: dict with:
      - 'context_left' : list[str]     # left context tokens (lemma or surface per flags)
      - 'target_terms' : list[str]     # target phrase tokens    (lemma or surface per flags)
      - 'context_right': list[str]     # right context tokens (lemma or surface per flags)
      - 'target_tokens': list[dict]    # raw token dicts of the target phrase (unchanged)

    Notes
    -----
    - If context_pos_filtered=True, only tokens with POS in pos_keep are included
      in context_left/context_right.
    - If context_lemmatized=True, context items use 'lemma' (falls back to 'token_text' if missing).
      Otherwise they use 'token_text'.
    - If target_lemmatized=True, target items use 'lemma' (falls back to 'token_text' if missing).
      Otherwise they use 'token_text'.
    """

    if not target_kwic_idx:
        return {
            "context_left": [],
            "target_terms": [],
            "context_right": [],
            "target_tokens": [],
        }

    # normalize & sort indices (allowing non-contiguous)
    idx = sorted(set(int(i) for i in target_kwic_idx))
    i0, i1 = idx[0], idx[-1]

    # slices for context; target collected by explicit indices so we don't assume contiguity
    left_tokens  = kwic_tokens[:i0]
    right_tokens = kwic_tokens[i1 + 1:]
    target_tokens = [kwic_tokens[i] for i in idx if 0 <= i < len(kwic_tokens)]

    def _tok_repr(t: dict, use_lemma: bool) -> str:
        if use_lemma:
            val = (t.get("lemma") or "").strip()
            return val if val else (t.get("token_text") or "")
        return t.get("token_text") or ""

    def _context_filter(tokens: list[dict]) -> list[dict]:
        if not context_pos_filtered:
            return tokens
        return [t for t in tokens if (t.get("pos") in pos_keep)]

    # build outputs
    left_filtered  = _context_filter(left_tokens)
    right_filtered = _context_filter(right_tokens)

    context_left  = [_tok_repr(t, context_lemmatized) for t in left_filtered]
    context_right = [_tok_repr(t, context_lemmatized) for t in right_filtered]
    target_terms  = [_tok_repr(t, target_lemmatized) for t in target_tokens]

    return {
        "context_left": context_left,
        "target_terms": target_terms,
        "context_right": context_right,
        "target_tokens": target_tokens,
    }

def augment_with_subwords_span(tokens, target_idx, tokenizer):
    """
    tokens:      list[dict] with 'token_text' and 'lemma'
    target_idx:  list[int]  (KWIC indices of the target tokens; contiguous)
    tokenizer:   HF-style tokenizer with encode/convert helpers

    Returns:
      sent_str         : str             # reconstructed (lowercased) sentence for encoding
      sp_tokens        : list[str]       # subword sequence incl. specials
      aug_tokens       : list[dict]      # tokens + {'sp_first','sp_pieces'}
      target_piece_span: (start, end)    # [start, end) span in subword space
    """
    aug_tokens = []
    words = []
    sp_tokens = []
    sp_pos = 0

    prepend = getattr(tokenizer, "cls_token", None) or "<s>"
    append  = getattr(tokenizer, "sep_token", None) or "</s>"

    if prepend:
        sp_tokens.append(prepend)
        sp_pos += 1

    # build sentence, piece offsets
    for t in tokens:
        word = (t["token_text"] or "").lower()
        try:
            word_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
        except Exception:
            word_ids = tokenizer.encode(word, add_special_tokens=False)
        subwords = tokenizer.convert_ids_to_tokens(word_ids)

        new_t = dict(t)
        new_t["sp_first"]  = sp_pos
        new_t["sp_pieces"] = subwords
        aug_tokens.append(new_t)

        words.append(word)
        sp_tokens.extend(subwords)
        sp_pos += len(subwords)

    if append:
        sp_tokens.append(append)

    sent_str = " ".join(words)

    # phrase piece span
    if not target_idx:
        target_piece_span = None
    else:
        i0 = min(target_idx)
        i1 = max(target_idx)
        t0 = aug_tokens[i0]
        t1 = aug_tokens[i1]
        start = t0["sp_first"]
        end   = t1["sp_first"] + len(t1["sp_pieces"])  # exclusive
        target_piece_span = (start, end)

    return sent_str, sp_tokens, aug_tokens, target_piece_span

import torch
import numpy as np
import inspect

def encode_trunc(text: str, tokenizer, device="cpu", max_len=512):
    kwargs = {"text": text, "return_tensors": "pt", "truncation": True, "max_length": max_len}
    if "add_special_tokens" in inspect.signature(tokenizer.__call__).parameters:
        kwargs["add_special_tokens"] = True
    result = tokenizer(**kwargs)
    return result.to(device) if hasattr(result, "to") else result

def hidden_phrase_embedding(
    sent_str: str,
    target_piece_span,  # (start, end)
    *,
    tokenizer,
    model,
    device="cpu",
    layer_idx=8,
    piece_pooling="mean",  # "mean"|"sum"|"max"
):
    if not target_piece_span:
        return np.zeros(model.config.hidden_size, dtype=np.float32)

    start, end = target_piece_span
    enc = encode_trunc(sent_str, tokenizer=tokenizer, device=device, max_len=getattr(tokenizer, "model_max_length", 512))
    enc["output_hidden_states"] = True

    with torch.no_grad():
        outs = model(**enc)
        H = outs.hidden_states[layer_idx].squeeze(0)  # [L, D]

    L = H.shape[0]
    if start >= L or end > L or start >= end:
        return np.zeros(model.config.hidden_size, dtype=np.float32)

    span = H[start:end]
    if piece_pooling == "sum":
        vec = span.sum(dim=0)
    elif piece_pooling == "max":
        vec = span.max(dim=0).values
    else:
        vec = span.mean(dim=0)
    return vec.cpu().numpy()

def embed_instance(
    row,
    tokenizer,
    model,
    *,
    device="cpu",
    layer_idx=11,
    piece_pooling="mean",
    context_lemmatized=True,
    target_lemmatized=False,
    context_pos_filtered=True,
    pos_keep=("NOUN", "ADJ", "VERB", "PROPN"),
):
    """
    Builds a contextual embedding for one KWIC instance (possibly multiword target phrase).

    Parameters
    ----------
    row : pd.Series or dict
        Must include 'kwic_tokens' (list of dicts) and 'target_kwic_idx' (list of ints).
    tokenizer : HuggingFace tokenizer or compatible
        Tokenizer used for Latin BERT.
    model : HuggingFace model
        Pretrained transformer model.
    device : str
        Device string ('cpu' or 'cuda').
    layer_idx : int
        Which hidden layer to use for embedding extraction.
    piece_pooling : str
        How to pool subword pieces ('mean', 'sum', or 'max').
    context_lemmatized : bool
        If True, use lemmas for context tokens; else use surface forms.
    target_lemmatized : bool
        If True, use lemmas for target tokens; else use surface forms.
    context_pos_filtered : bool
        If True, include only tokens with POS in pos_keep in context.
    pos_keep : tuple[str]
        POS tags to include when filtering context tokens.

    Returns
    -------
    dict with standardized keys:
      - "embedding"        : np.ndarray  (vector representation of target phrase)
      - "target_terms"     : list[str]
      - "context_left"     : list[str]
      - "context_right"    : list[str]
      - "target_tokens"    : list[dict]
      - "grela_id"         : str
      - "target_sentence_id": str
      - "sentence_text"    : str
      - "span_subwords"    : tuple[int, int]  (subword start/end span)
    """

    kwic_tokens = row["kwic_tokens"]
    target_idx = row["target_kwic_idx"]

    # --- (1) Context split with configurable lemmatization/POS filtering ---
    ctx_data = process_kwic_tokens(
        kwic_tokens,
        target_idx,
        context_lemmatized=context_lemmatized,
        target_lemmatized=target_lemmatized,
        context_pos_filtered=context_pos_filtered,
        pos_keep=pos_keep,
    )

    # --- (2) Subword augmentation (target span) ---
    sent_str, sp_toks, aug_toks, span = augment_with_subwords_span(
        kwic_tokens,
        target_idx,
        tokenizer,
    )

    # --- (3) Hidden-state pooling for the target phrase ---
    vec = hidden_phrase_embedding(
        sent_str,
        span,
        tokenizer=tokenizer,
        model=model,
        device=device,
        layer_idx=layer_idx,
        piece_pooling=piece_pooling,
    )

    # --- (4) Collect outputs ---
    return {
        "embedding": vec,
        "target_terms": ctx_data["target_terms"],
        "context_left": ctx_data["context_left"],
        "context_right": ctx_data["context_right"],
        "target_tokens": ctx_data["target_tokens"],
        "grela_id": row.get("grela_id"),
        "target_sentence_id": row.get("target_sentence_id"),
        "sentence_text": row.get("target_sentence_text"),
        "span_subwords": span,
    }