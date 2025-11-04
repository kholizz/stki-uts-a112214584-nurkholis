from src.search import run_vsm

def generate_response(query, data_dir="data/processed", k=3):
    rows, explain = run_vsm(query, data_dir, k=k)
    keys = [f"[{d}]" for d,_,_ in rows]
    answer = f"Dokumen terkait: {'; '.join(keys)}"
    return answer, rows, explain

if __name__ == "__main__":
    print("Mini Search (ketik 'exit' untuk keluar)")
    while True:
        q = input("Query: ").strip()
        if q.lower() == "exit":
            break
        ans, rows, exp = generate_response(q)
        print(ans)
        for d, s, sn in rows:
            print(f"- {d} (cosine={s}) :: {sn}")
