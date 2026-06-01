from collections import defaultdict
import json
import os
import numpy as np

def acc(ls):
    return np.sum(ls) / len(ls) if len(ls) else 0.0

def eval(outs, model_name, result_dir="pred_results", split="val", task_name=None):
    os.makedirs(result_dir, exist_ok=True)

    result_bin = defaultdict(list)
    pred_results = []

    count, correct = 0, 0
    for out in outs:
        image_ids = out.get("image_id", [None] * len(out["pred"]))
        for pred, gt, n_obj_exist, image_id in zip(out["pred"], out["gt"], out["n_obj_exist"], image_ids):
            
            count += 1
            if ("Yes" in gt and "yes" in pred.lower()) or ("No" in gt and "no" in pred.lower()): 
                
                correct += 1
                result_bin[n_obj_exist].append(1)

            else:
                result_bin[n_obj_exist].append(0)

            pred_results.append({
                "pred": pred,
                "gt": gt,
                "image_id": image_id,
                "n_obj_exist": n_obj_exist
            })

    accuracy = 1.0 * correct / count if count else 0.0
    print("accuracy: {:.4f}".format(accuracy))

    # 1 - 10; 10 - 20; > 20
    bins = defaultdict(list)
    for k, v in result_bin.items():
        if k < 10:
            bins[0].extend(result_bin[k])
        elif k < 20:
            bins[1].extend(result_bin[k])
        else:
            bins[2].extend(result_bin[k])

    metrics = {
        "accuracy": accuracy,
        "1 - 9": float(acc(bins[0])),
        "10 - 19": float(acc(bins[1])),
        ">= 20": float(acc(bins[2])),
    }

    task_prefix = f"{task_name}_" if task_name else ""
    save_filename = f"multiclass_result_{task_prefix}{split}_{model_name}"
    result_file = os.path.join(result_dir, f"{save_filename}.json")
    metrics_file = os.path.join(result_dir, f"{save_filename}_metrics.json")
    with open(result_file, "w", encoding="utf-8") as fp:
        json.dump(pred_results, fp, indent=2)
    with open(metrics_file, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    print("result file saved to %s" % result_file)
    print("metrics file saved to %s" % metrics_file)

    print("1 - 9: {:.4f}".format(metrics["1 - 9"]))
    print("10 - 19: {:.4f}".format(metrics["10 - 19"]))
    print(">= 20: {:.4f}".format(metrics[">= 20"]))

    return metrics
