def hit_rate_at_k(gt_ids, retrieved_ids):
    return int(len(set(gt_ids) & set(retrieved_ids)) > 0)

def recall_at_k(gt_ids, retrieved_ids):
    gt = set(gt_ids)
    retrieved = set(retrieved_ids)
    return len(gt & retrieved) / len(gt) if gt else 0

def precision_at_k(gt_ids, retrieved_ids):
    gt = set(gt_ids)
    retrieved = set(retrieved_ids)
    return len(gt & retrieved) / len(retrieved) if retrieved else 0
