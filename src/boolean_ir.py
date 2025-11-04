from collections import defaultdict

def load_processed_docs(dir_path):
    import os
    docs = {}
    for fname in sorted(os.listdir(dir_path)):
        if fname.endswith('.txt'):
            with open(os.path.join(dir_path, fname), 'r', encoding='utf-8') as f:
                docs[fname] = f.read().split()
    return docs

def build_inverted_index(docs):
    inv = defaultdict(set)
    for doc_id, tokens in docs.items():
        for t in set(tokens):
            inv[t].add(doc_id)
    return inv

def build_incidence_matrix(docs):
    vocab = sorted({t for toks in docs.values() for t in toks})
    doc_ids = sorted(docs.keys())
    index = { (di, t): 0 for di in doc_ids for t in vocab }
    for di in doc_ids:
        tokens = set(docs[di])
        for t in tokens:
            index[(di, t)] = 1
    return vocab, doc_ids, index

def eval_bool_query(inv_idx, all_docs, query: str):
    tokens = query.split()
    seq = []
    for tok in tokens:
        if tok in ("AND","OR","NOT"):
            seq.append(tok)
        else:
            seq.append(inv_idx.get(tok, set()))

    # NOT
    i = 0
    while i < len(seq):
        if seq[i] == "NOT" and i+1 < len(seq) and isinstance(seq[i+1], set):
            seq[i:i+2] = [set(all_docs) - seq[i+1]]
        else:
            i += 1

    # AND
    i = 0
    while i < len(seq):
        if i+2 < len(seq) and isinstance(seq[i], set) and seq[i+1] == "AND" and isinstance(seq[i+2], set):
            seq[i:i+3] = [seq[i] & seq[i+2]]
        else:
            i += 1

    # OR
    i = 0
    while i < len(seq):
        if i+2 < len(seq) and isinstance(seq[i], set) and seq[i+1] == "OR" and isinstance(seq[i+2], set):
            seq[i:i+3] = [seq[i] | seq[i+2]]
        else:
            i += 1

    result = seq[0] if seq and isinstance(seq[0], set) else set()
    explain = f"Evaluated: {query}"
    return sorted(result), explain
