import json
import sys
import os

SAVE_DIR = "comparison_results"

def load_results(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)["individual_results"]

def load_acc(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return float(data["exact_match_accuracy"])

def main(file1_path, file2_path):
    file1_name = os.path.basename(file1_path).replace(".json", "")
    file2_name = os.path.basename(file2_path).replace(".json", "")

    file1_acc = load_acc(file1_path)
    file1_acc = round(load_acc(file1_path), 2)
    file2_acc = round(load_acc(file2_path), 2)

    results1 = {item["id"]: item for item in load_results(file1_path)}
    results2 = {item["id"]: item for item in load_results(file2_path)}

    output = {
        "accuracy": "{file1_acc} vs {file2_acc}".format(file1_acc=file1_acc, file2_acc=file2_acc),
        "file1_correct_file2_incorrect": [],
        "file2_correct_file1_incorrect": [],
        "both_correct": [],
        "both_incorrect": [],
    }

    for qid in results1:
        item1 = results1[qid]
        item2 = results2.get(qid)

        if item2 is None:
            continue  # Skip unmatched IDs

        em1 = item1["exact_match"]
        em2 = item2["exact_match"]

        entry = {
            "id": qid,
            "question": item1["question"],
            "reference_answer": item1["reference_answer"],
            f"{file1_name}_predicted_answer": item1["predicted_answer"],
            f"{file2_name}_predicted_answer": item2["predicted_answer"],
            "url": item1["reference_url"]
        }

        if em1 == 1 and em2 == 0:
            output["file1_correct_file2_incorrect"].append(entry)
        elif em1 == 0 and em2 == 1:
            output["file2_correct_file1_incorrect"].append(entry)
        elif em1 == 1 and em2 == 1:
            output["both_correct"].append(entry)
        elif em1 == 0 and em2 == 0:
            output["both_incorrect"].append(entry)

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    output_file = os.path.join(SAVE_DIR, f"{file1_name}__{file2_name}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("比較完成，已輸出為 comparison_result.json")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_em.py <file1.json> <file2.json>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
