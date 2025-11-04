import math

def precision(pred, gold):
    pred = list(pred)
    gold = set(gold)
    tp = sum(1 for d in pred if d in gold)
    return tp / max(len(pred), 1)

def recall(pred, gold):
    pred = list(pred)
    gold = set(gold)
    tp = sum(1 for d in pred if d in gold)
    return tp / max(len(gold), 1)

def f1(p, r):
    return 2*p*r / max(p+r, 1e-12)

def precision_at_k(ranked, gold, k):
    return precision([d for d,_ in ranked[:k]], gold)

def average_precision(ranked, gold):
    gold = set(gold)
    score, hit = 0.0, 0
    for i, (d, _) in enumerate(ranked, 1):
        if d in gold:
            hit += 1
            score += hit / i
    return score / max(len(gold), 1)

def map_at_k(list_of_ranked, list_of_gold, k):
    aps = []
    for ranked, gold in zip(list_of_ranked, list_of_gold):
        aps.append(average_precision(ranked[:k], gold))
    return sum(aps)/max(len(aps),1)

def dcg_at_k(ranked, gold, k):
    gold = set(gold)
    dcg = 0.0
    for i, (d, _) in enumerate(ranked[:k], 1):
        rel = 1.0 if d in gold else 0.0
        dcg += (2**rel - 1) / math.log2(i+1)
    return dcg

def ndcg_at_k(ranked, gold, k):
    ideal = [(d,1.0) for d,_ in ranked if d in set(gold)]
    dcg = dcg_at_k(ranked, gold, k)
    idcg = dcg_at_k(ideal, [d for d,_ in ideal], k) if ideal else 1.0
    return dcg / idcg if idcg > 0 else 0.0
