import argparse, os
from src.preprocess import preprocess_text
from src.boolean_ir import load_processed_docs, build_inverted_index, eval_bool_query
from src.vsm_ir import build_tfidf_index, rank_docs

def snippet(doc_path, doc_id, length=120):
    with open(os.path.join(doc_path, doc_id), 'r', encoding='utf-8') as f:
        s = f.read()
    return (s[:length] + '...') if len(s) > length else s

def run_boolean(query, docs_processed_dir):
    docs = load_processed_docs(docs_processed_dir)
    inv = build_inverted_index(docs)
    all_docs = set(docs.keys())
    result, explain = eval_bool_query(inv, all_docs, query)
    rows = [(d, 1.0, snippet(docs_processed_dir, d)) for d in result]
    return rows, explain

def run_vsm(query, docs_processed_dir, k=5, sublinear=False):
    docs = load_processed_docs(docs_processed_dir)
    doc_vecs, idf = build_tfidf_index(docs, sublinear=sublinear)
    from src.preprocess import preprocess_text
    q_tokens = preprocess_text(query)
    ranked = rank_docs(q_tokens, doc_vecs, idf, k=k, sublinear=sublinear)
    rows = [(d, float(f"{s:.6f}"), snippet(docs_processed_dir, d)) for d, s in ranked]
    explain = "Top-k dihitung menggunakan cosine similarity pada ruang TF-IDF"
    return rows, explain

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["boolean","vsm"], required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--sublinear", action="store_true", help="aktifkan sublinear TF untuk TF-IDF")
    ap.add_argument("--data", default="data/processed")
    args = ap.parse_args()

    if args.model == "boolean":
        rows, explain = run_boolean(args.query, args.data)
    else:
        rows, explain = run_vsm(args.query, args.data, k=args.k, sublinear=args.sublinear)

    print("doc_id\tcosine/score\tsnippet")
    for d, s, snip in rows:
        print(f"{d}\t{s}\t{snip}")
    print("\nExplain:", explain)

if __name__ == "__main__":
    main()
