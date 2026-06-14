from tqdm import tqdm
from datasets import load_dataset
from .utils import get_model_device, save_features, load_features
import os
import torch


def text_correct(result):
    return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]


def image_correct(result):
    return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]


def group_correct(result):
    return image_correct(result) and text_correct(result)


def compute_clip_scores(logits_per_text):
    clip_score_c0_i0 = logits_per_text[0, 0].item()
    clip_score_c1_i0 = logits_per_text[1, 0].item()
    clip_score_c0_i1 = logits_per_text[0, 1].item()
    clip_score_c1_i1 = logits_per_text[1, 1].item()
    return clip_score_c0_i0, clip_score_c1_i0, clip_score_c0_i1, clip_score_c1_i1


def process_example(model, example, device):
    text_list = [example["caption_0"], example["caption_1"]]
    text_tokens = model.text_model.tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    image0 = example["image_0"].convert("RGB")  
    image1 = example["image_1"].convert("RGB")
    images = model.vision_model.image_processor(
        [image0, image1], return_tensors="pt"
    ).to(device)
    with torch.amp.autocast(device_type='cuda'):
        with torch.no_grad():
            outputs = model.forward(images, text_tokens, text_list = text_list,  return_encoded=True)
    encoded_image_features = outputs["encoded_image_features"]
    encoded_text_features = outputs["encoded_text_features"]
    logits_per_text = outputs["logits_per_text"]

    clip_scores = compute_clip_scores(logits_per_text)
    return encoded_image_features.cpu(), encoded_text_features.cpu(), clip_scores


def evaluate_clip_scores(
    model, pre_encode_image_features, pre_encode_text_features, device
):
    winoground_clip_scores = []

    for key, encode_image_features in pre_encode_image_features.items():
        encode_image_features = encode_image_features.to(device)
        encode_text_features = pre_encode_text_features[key].to(device)
        with torch.amp.autocast(device_type='cuda'):
            with torch.no_grad():
                outputs = model.forward(
                encode_image_features, encode_text_features, is_pre_encoded=True
            )
        logits_per_text = outputs["logits_per_text"]

        clip_scores = compute_clip_scores(logits_per_text)
        winoground_clip_scores.append(
            {
                "id": key,
                "c0_i0": clip_scores[0],
                "c0_i1": clip_scores[2],
                "c1_i0": clip_scores[1],
                "c1_i1": clip_scores[3],
            }
        )

    return winoground_clip_scores


def winoground_eval(model, text_model_name, vision_model_name, save_dir):
    auth_token = os.getenv("HF_AUTH_TOKEN")
    device = get_model_device(model)
    winoground_clip_scores = []

    image_feature_path = f"{save_dir}/{vision_model_name}/winoground.pt"
    text_feature_path = f"{save_dir}/{text_model_name}/winoground.pt"

    if not os.path.exists(image_feature_path) or not os.path.exists(text_feature_path):
        winoground = load_dataset("facebook/winoground", use_auth_token=auth_token)[
            "test"
        ]
        pre_encode_image_features = {}
        pre_encode_text_features = {}

        for example in tqdm(winoground):
            encoded_image_features, encoded_text_features, clip_scores = (
                process_example(model, example, device)
            )
            pre_encode_image_features[example["id"]] = encoded_image_features
            pre_encode_text_features[example["id"]] = encoded_text_features
            winoground_clip_scores.append(
                {
                    "id": example["id"],
                    "c0_i0": clip_scores[0],
                    "c0_i1": clip_scores[2],
                    "c1_i0": clip_scores[1],
                    "c1_i1": clip_scores[3],
                }
            )

        save_features(pre_encode_image_features, image_feature_path)
        save_features(pre_encode_text_features, text_feature_path)
    else:
        pre_encode_image_features = load_features(image_feature_path)
        pre_encode_text_features = load_features(text_feature_path)
        winoground_clip_scores = evaluate_clip_scores(
            model, pre_encode_image_features, pre_encode_text_features, device
        )

    text_correct_count = sum(
        1 for result in winoground_clip_scores if text_correct(result)
    )
    image_correct_count = sum(
        1 for result in winoground_clip_scores if image_correct(result)
    )
    group_correct_count = sum(
        1 for result in winoground_clip_scores if group_correct(result)
    )

    denominator = len(winoground_clip_scores)
    text_score = text_correct_count / denominator
    image_score = image_correct_count / denominator
    group_score = group_correct_count / denominator

    print("text score:", text_score)
    print("image score:", image_score)
    print("group score:", group_score)

    return {
        "text": text_score,
        "image": image_score,
        "group": group_score,
    }
