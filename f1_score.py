#!/usr/bin/env python3
# compute_f1.py

import json
import string
from collections import Counter

def normalize_answer(s: str) -> str:
    """小寫、去標點、去多餘空格"""
    s = s.lower().strip()
    # 去除標點
    return ''.join(ch for ch in s if ch not in set(string.punctuation)).strip()

def f1_score(predicted: str, reference: str) -> float:
    """計算 token-level F1"""
    pred_tokens = normalize_answer(predicted).split()
    ref_tokens  = normalize_answer(reference).split()
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0 or not pred_tokens or not ref_tokens:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)

def main(src_file, tgt_file):
    # 1. 讀檔
    with open(src_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 針對 individual_results 計算每題 F1，並累加
    f1_list = []
    for item in data.get('individual_results', []):
        pred = item.get('predicted_answer', '')
        ref  = item.get('reference_answer', '')
        f1   = f1_score(pred, ref)
        f1_list.append(f1)

    # 3. 計算平均 F1
    avg_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0

    # 4. 把 avg_f1 加到最外層
    data = {'f1_score': avg_f1, **data}

    # 5. 輸出新的 JSON
    with open(tgt_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"已寫入 evaluation_results_with_f1.json，整體 F1 = {avg_f1:.4f}")

if __name__ == '__main__':
    src_file = 'eval/evaluation_results_rewrite_rerank_multi_retrieve.json'
    tgt_file = 'eval/evaluation_results_rewrite_rerank_multi_retrieve.json'
    main(src_file, tgt_file)
