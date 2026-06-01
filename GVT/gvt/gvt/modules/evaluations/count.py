import os
import re
import json
import numpy as np
from collections import defaultdict, OrderedDict

from gvt.modules.evaluations.dist import get_rank, get_world_size

str2d = {
    "none": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nive": 9
}


def save_result(result, result_dir, filename, remove_duplicate=""):
    import json

    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(
        result_dir, "%s_rank%d.json" % (filename, get_rank())
    )
    final_result_file = os.path.join(result_dir, "%s.json" % filename)

    json.dump(result, open(result_file, "w"))


    result = []
    for rank in range(get_world_size()):
        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, rank)
        )
        res = json.load(open(result_file, "r"))
        result += res

    if remove_duplicate:
        result_new = []
        id_list = []
        for res in result:
            if res[remove_duplicate] not in id_list:
                id_list.append(res[remove_duplicate])
                result_new.append(res)
        result = result_new

    json.dump(result, open(final_result_file, "w"))
    print("result file saved to %s" % final_result_file)

    return final_result_file


def extract_digit(string):
    for k, v in str2d.items():
        string = string.replace(k, str(v))

    digit = re.findall("\d+", string)
    if digit and len(digit):
        digit = int(digit[0])
    else:
        digit = 0
    return digit


def _report_metrics(result_file):
    count, correct, mae_sum, mse_sum = 0, 0, 0, 0

    pred_gt = json.load(open(result_file))

    result_bin = defaultdict(list)
    for entry in pred_gt:
        pred = extract_digit(entry["pred"])
        gt = entry["gt"]

        pred = int(pred)
        gt = int(gt)

        count += 1
        if int(pred) == int(gt):
            correct += 1

        mae_sum += np.abs(int(gt) - int(pred))
        mse_sum += np.power(int(gt) - int(pred), 2) 

        result_bin[gt//3].append([pred, gt])


    return {
        "accuracy": correct / count, 
        "mae": mae_sum / count,
        "rmse": np.sqrt(mse_sum / count),
    }, result_bin


def stat(result):
    count = len(result)
    mae_sum = 0
    mse_sum = 0
    correct = 0
    for pred, gt in result:
        pred = int(pred)
        gt = int(gt)
        
        mae_sum += np.abs(int(gt) - int(pred))
        mse_sum += np.power(int(gt) - int(pred), 2)
        if pred == gt: 
            correct += 1

    return{
        "acc":  "{:.4f}".format(1.0 * correct / count),
        "mae":  "{:.4f}".format(mae_sum / count),
        "rmse": "{:.4f}".format(np.sqrt(mse_sum / count))
    }


def eval(outputs, model_name, split="val", result_dir="pred_results/count", task_name=None):
    pred_results = []
    for out in outputs:
        for pred, gt, image_id, n_obj_exist in zip(out['pred'], out['gt'], out['image_id'], out['n_obj_exist']):
            pred_results.append({
                "pred": pred,
                "gt": gt,
                "image_id": image_id,
                "n_obj_exist": n_obj_exist
            })

    task_prefix = f"{task_name}_" if task_name else ""
    save_filename = f"count_result_{task_prefix}{split}_{model_name}"
    result_file = save_result(
        pred_results,
        result_dir=result_dir,
        filename=save_filename
    )

    metrics, result_bin = _report_metrics(result_file)
    metrics_file = os.path.join(result_dir, f"{save_filename}_metrics.json")
    if get_rank() == 0:
        with open(metrics_file, "w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
        print("metrics file saved to %s" % metrics_file)

    for k, v in metrics.items():
        print(k, "{:.4f}".format(v))

    keys = sorted(list(result_bin.keys()))
    new_result_bin = OrderedDict({k: result_bin[k] for k in keys})

    for k, v in new_result_bin.items():
        print(f"{k*3}~{(k+1)*3}:", stat(v))

    return metrics
