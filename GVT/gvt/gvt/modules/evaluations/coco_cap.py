import os
import json
import shutil

from pycocotools.coco import COCO

from gvt.modules.evaluations.cider.cider import Cider
from gvt.modules.evaluations.spice.spice import Spice
from gvt.modules.evaluations.tokenizer.ptbtokenizer import PTBTokenizer
from gvt.modules.evaluations.dist import get_rank, get_world_size


def check_java_eval_dependencies():
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    required_files = [
        os.path.join(eval_dir, "tokenizer", "stanford-corenlp-3.4.1.jar"),
        os.path.join(eval_dir, "spice", "spice-1.0.jar"),
        os.path.join(eval_dir, "spice", "lib", "ejml-0.23.jar"),
        os.path.join(eval_dir, "spice", "lib", "slf4j-api-1.7.12.jar"),
        os.path.join(eval_dir, "spice", "lib", "slf4j-simple-1.7.21.jar"),
        os.path.join(eval_dir, "spice", "lib", "lmdbjni-0.4.6.jar"),
        os.path.join(eval_dir, "spice", "lib", "lmdbjni-linux64-0.4.6.jar"),
        os.path.join(eval_dir, "spice", "lib", "lmdbjni-osx64-0.4.6.jar"),
        os.path.join(eval_dir, "spice", "lib", "lmdbjni-win64-0.4.6.jar"),
        os.path.join(eval_dir, "spice", "lib", "fst-2.47.jar"),
        os.path.join(eval_dir, "spice", "lib", "jackson-core-2.5.3.jar"),
        os.path.join(eval_dir, "spice", "lib", "javassist-3.19.0-GA.jar"),
        os.path.join(eval_dir, "spice", "lib", "objenesis-2.4.jar"),
        os.path.join(eval_dir, "spice", "lib", "guava-19.0.jar"),
        os.path.join(eval_dir, "spice", "lib", "json-simple-1.1.1.jar"),
        os.path.join(eval_dir, "spice", "lib", "Meteor-1.5.jar"),
        os.path.join(eval_dir, "spice", "lib", "SceneGraphParser-1.0.jar"),
        os.path.join(eval_dir, "spice", "lib", "stanford-corenlp-3.6.0.jar"),
        os.path.join(eval_dir, "spice", "lib", "stanford-corenlp-3.6.0-models.jar"),
    ]
    missing_files = [path for path in required_files if not os.path.isfile(path)]
    if shutil.which("java") is None:
        raise RuntimeError(
            "COCO Caption evaluation requires Java for PTBTokenizer/SPICE. "
            "Install Java first, e.g. `conda install -c conda-forge openjdk=8`."
        )
    if missing_files:
        missing = "\n  ".join(missing_files)
        raise RuntimeError(
            "COCO Caption evaluation is missing Java jar dependencies:\n  "
            f"{missing}\n`scripts/run_all_eval.sh` prepares these automatically for caption tasks. "
            "If you call this evaluator directly, run `GVT/gvt/scripts/download_eval_deps.sh` first."
        )


class COCOEvalCap:
    def __init__(self, coco, cocoRes):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': cocoRes.getImgIds()}

    def evaluate(self):
        imgIds = self.params['image_id']

        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Cider(), "CIDEr"),
            (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]


def save_result(result, result_dir, filename, remove_duplicate=""):

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


def coco_caption_eval(coco_gt_root, results_file, split):
    check_java_eval_dependencies()

    filenames = {
        "val": "coco_karpathy_val_gt.json",
        "test": "coco_karpathy_test_gt.json",
    }

    annotation_file = os.path.join(coco_gt_root, filenames[split])

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    return coco_eval


def _report_metrics(eval_result_file, split_name, coco_gt_root=None):

    if coco_gt_root is None or not os.path.isdir(coco_gt_root):
        coco_gt_root = "eval_gt"
    coco_val = coco_caption_eval(coco_gt_root, eval_result_file, split_name)
    return coco_val


def eval(outs, model_name, coco_gt_root=None, result_dir="pred_results", split="val"):
    results = []
    for out in outs:
        captions = out['pred']
        iids = out["image_id"]
        for iid, caption in zip(iids, captions):
            results.append({"caption": caption,
                            "id": iid,
                            "image_id": iid})

    new_results =   []
    for item in results:
        if 'id' not in item:
            item['id'] = item['image_id']
        if 'caption' not in item:
            item['caption'] = item['caption:']
        new_results.append(item)

    result_file = save_result(
        result=new_results,
        result_dir=result_dir,
        filename=f"caption_result_{split}_{model_name}",
        remove_duplicate="id",
    )

    metrics = _report_metrics(result_file, split_name=split, coco_gt_root=coco_gt_root)
    if get_rank() == 0:
        metrics_file = os.path.join(result_dir, f"caption_result_{split}_{model_name}_metrics.json")
        with open(metrics_file, "w", encoding="utf-8") as fp:
            json.dump(metrics.eval, fp, indent=2)
        print("metrics file saved to %s" % metrics_file)

    print("metrics:")
    print(metrics)
    return metrics
