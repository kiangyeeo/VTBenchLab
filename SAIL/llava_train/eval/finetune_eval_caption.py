import argparse
import torch
import os
import sys

# these two lines are necessary for computing the Caption Score, please download the COCOEvalCap from https://github.com/sks3i/pycocoevalcap and import COCOEvalCap
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

import json
from tqdm import tqdm
import shortuuid

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from llava.model.builder import load_pretrained_model
from llava.model import *
import transformers
from PIL import Image
import math
from tqdm import tqdm


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_question(question_file):
    data = json.load(open(question_file))
    if "coco" in question_file:
        annotations = data["annotations"]
        # select 1000 images
        selected_data = []
        image_ids = []
        for entry in annotations:
            if entry["image_id"] not in image_ids:
                image_ids.append(entry["image_id"])
                selected_data.append(entry)
            else:
                pass
            if len(image_ids) == 1000:
                break
        images = data["images"]
        selected_images = {}
        for image in images:
            if image["id"] in image_ids:
                selected_images[image["id"]] = image
                if len(selected_images.keys()) == 1000:
                    break
        return selected_data, selected_images
    else:
        return data, {}


def prune_question_file(question_file):
    final_data = {}
    data = json.load(open(question_file))

    for key in ["annotations", "images"]:
        final_data[key] = data[key]

    return final_data


def eval_model(parser):

    args = parser.parse_args()
    questions, images_dict = get_question(args.question_file)
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )
    model.eval()

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    gt_file = os.path.expanduser(args.gt_file)
    os.makedirs(os.path.dirname(gt_file), exist_ok=True)
    gt_file = open(gt_file, "w")

    qs_temp = "<image>\nWhat is this image about? Describe it in a sentence with no more than 20 words.\n"
    eval_input = []
    is_coco = "coco" in args.question_file
    eval_gt = []
    images = []
    for i, line in tqdm(enumerate(questions), total=len(questions)):
        idx = line["id"]
        if is_coco:
            image_id = str(line["image_id"]).zfill(12) + ".jpg"
        else:
            image_id = line["image"]

        qs = qs_temp

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        image = Image.open(os.path.join(args.image_folder, image_id))
        image_tensor = (
            image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            .unsqueeze(0)
            .half()
            .cuda()
        )
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=1024,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()

        tmp_dict = {}
        gt_dict = {}

        tmp_dict["image_id"] = line["image_id"]
        tmp_dict["caption"] = outputs

        gt_dict["id"] = idx
        gt_dict["image_id"] = line["image_id"]
        gt_dict["caption"] = line["caption"]
        eval_input.append(tmp_dict)
        eval_gt.append(gt_dict)
        images.append(images_dict[line["image_id"]])

    ans_file.write(json.dumps(eval_input, indent=4))
    ans_file.flush()
    gt_file.write(json.dumps({"images": images, "annotations": eval_gt}, indent=4))
    gt_file.flush()

    coco = COCO(args.gt_file)
    coco_result = coco.loadRes(args.answers_file)
    COCO_eval = COCOEvalCap(coco, coco_result)
    score_dict = COCO_eval.evaluate()
    print(score_dict)
    # ans_file.close()


if __name__ == "__main__":
    # Load model directly

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_base",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/network/scratch/q/qian.yang/visual_imagination/llava_stage1_clipscore_filtered_100K_InternVL_Recap",
    )

    parser.add_argument(
        "--image-folder",
        type=str,
        default="/home/mila/q/qian.yang/scratch/coco2017/2017/val2017",
    )
    parser.add_argument(
        "--question-file",
        type=str,
        default="/home/mila/q/qian.yang/scratch/coco2017/2017/annotations/annotations/captions_val2017.json",
    )
    parser.add_argument(
        "--answers-file",
        type=str,
        default="/network/scratch/q/qian.yang/visual_imagination/llava_stage1_clipscore_filtered_100K_InternVL_Recap/pretrain_coco2017_1k_eval.json",
    )
    parser.add_argument(
        "--gt-file",
        type=str,
        default="/network/scratch/q/qian.yang/visual_imagination/llava_stage1_clipscore_filtered_100K_InternVL_Recap/pretrain_coco2017_1k_eval_gt.json",
    )
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")

    eval_model(parser)
