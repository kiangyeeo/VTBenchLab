import torch
from .coco_dataset import CocoCaptions
import torch.utils.data as dutils
from typing import List
import os
from tqdm import tqdm
import torch.nn as nn
from .utils import get_model_device, save_features, load_features
from torch.cuda.amp import autocast


class Processor:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, images):
        outputs = self.processor(images, return_tensors="pt")
        if isinstance(outputs, torch.Tensor):
            return outputs
        else:
            return outputs['pixel_values']

def coco_collate_fn(batch):
    text_list = []
    image_list = []
    index_list = []
    for item in batch:
        image, text, index = item
        text_list.append(text)
        image_list.append(image)
        index_list.append(index)
    # print(image_list)
    images = torch.cat(image_list)
    # print(images.shape)
    images = {"pixel_values": images}
    return text_list, images, index_list


# Encodes all text and images in a dataset
def encode_dataset(
    model,
    dataset: dutils.Dataset,
    device,
    batch_size,
    text_model_name: str,
    vision_model_name: str,
    save_dir: str = None,
):
    save_backbone_image_features_path = os.path.join(
        save_dir, f"{vision_model_name}/coco.pt"
    )
    save_backbone_text_features_path = os.path.join(
        save_dir, f"{text_model_name}/coco.pt"
    )
    with torch.no_grad():
        # image_to_text_map[i] gives the corresponding text indices for the ith image
        #  (as there are multiple pieces of text for each image)
        image_to_text_map = []

        # text_to_image_map[i] gives the corresponding image index for the ith text
        text_to_image_map = []

        image_encodings = []
        text_encodings = []

        text_index = 0
        image_index = 0
        captions_per_image = 5

        if not os.path.exists(save_backbone_image_features_path) or not os.path.exists(
            save_backbone_text_features_path
        ):
            dataloader = dutils.DataLoader(
                dataset,
                collate_fn=coco_collate_fn,
                batch_size=batch_size,
                shuffle=False,
            )
            pre_encode_image_features = {}
            pre_encode_text_features = {}
            for text, images, indexs in tqdm(dataloader):
                images = {
                    key: value.to(device) for key, value in images.items()
                }  # B x 3 x 224 x 224
                batch_size = len(text)
                text_list = []
                for i in text:
                    text_list.extend(i[:captions_per_image])
                text_tokens = model.text_model.tokenizer(
                    text_list, padding=True, truncation=True, return_tensors="pt"
                ).to(
                    device
                )  # (B*5) x 77
                # Update text_to_image_map and image_to_text_map for this batch
                for i in range(batch_size):
                    # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
                    text_indices = list(
                        range(text_index, text_index + captions_per_image)
                    )
                    image_to_text_map.append(text_indices)
                    text_index += captions_per_image
                    # Each of the next captions_per_image text captions correspond to the same image
                    text_to_image_map += [image_index] * captions_per_image
                    image_index += 1
                image_features, encoded_image_features = model.encode_image(
                    images, return_encoded=True
                )
                image_encodings.append(image_features)
                text_features, encoded_text_features = model.encode_text(
                    text_tokens, text_list=text_list, return_encoded=True
                )
                text_encodings.append(text_features)
                for i, index in enumerate(indexs):
                    pre_encode_image_features[index] = encoded_image_features[i].cpu()
                    # for text we have 5 captions per image
                    pre_encode_text_features[index] = encoded_text_features[
                        i * 5 : (i + 1) * 5
                    ].cpu()
            save_features(pre_encode_image_features, save_backbone_image_features_path)
            save_features(pre_encode_text_features, save_backbone_text_features_path)
        else:
            pre_encode_image_features = load_features(save_backbone_image_features_path)
            pre_encode_text_features = load_features(save_backbone_text_features_path)
            batched_pre_encode_image_features = {}
            for i, (key, value) in enumerate(pre_encode_image_features.items()):
                if i % batch_size == 0:
                    batched_pre_encode_image_features[i // batch_size] = {}
                batched_pre_encode_image_features[i // batch_size][key] = value
            for i, batch in tqdm(batched_pre_encode_image_features.items()):
                encoded_image_features = []
                encoded_text_features = []
                for key, value in batch.items():
                    encoded_image_features.append(value)
                    encoded_text_features.append(pre_encode_text_features[key])
                encoded_image_features = torch.stack(encoded_image_features).to(device)
                image_features = model.encode_image(
                    encoded_image_features, is_pre_encoded=True
                )
                image_encodings.append(image_features)
                encoded_text_features = torch.stack(encoded_text_features).to(device)
                # resize
                encoded_text_features = encoded_text_features.view(
                    -1, encoded_text_features.shape[-1]
                )
                text_features = model.encode_text(
                    encoded_text_features, is_pre_encoded=True
                )
                text_encodings.append(text_features)
                batch_size = len(image_features)
                for i in range(batch_size):
                    # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
                    text_indices = list(
                        range(text_index, text_index + captions_per_image)
                    )
                    image_to_text_map.append(text_indices)
                    text_index += captions_per_image
                    # Each of the next captions_per_image text captions correspond to the same image
                    text_to_image_map += [image_index] * captions_per_image
                    image_index += 1

        image_encodings = torch.cat(image_encodings)
        text_encodings = torch.cat(text_encodings)
        text_to_image_map = torch.LongTensor(text_to_image_map).to(device)
        image_to_text_map = torch.LongTensor(image_to_text_map).to(device)

        # Normalise encodings
        image_encodings = image_encodings / image_encodings.norm(dim=-1, keepdim=True)
        text_encodings = text_encodings / text_encodings.norm(dim=-1, keepdim=True)

        return image_encodings, text_encodings, text_to_image_map, image_to_text_map


def recall_at_k(
    clip,
    dataset: dutils.Dataset,
    device,
    k_vals: List[int],
    batch_size: int,
    text_model_name: str,
    vision_model_name: str,
    save_dir: str = None,
):
    print("Encoding all data...")
    image_encodings, text_encodings, text_to_image_map, image_to_text_map = (
        encode_dataset(
            clip,
            dataset,
            device,
            batch_size=batch_size,
            text_model_name=text_model_name,
            vision_model_name=vision_model_name,
            save_dir=save_dir,
        )
    )

    num_text = text_encodings.shape[0]
    num_im = image_encodings.shape[0]
    captions_per_image = image_to_text_map.shape[1]

    # text-to-image recall
    print("Text-to-image recall...")

    dist_matrix = (
        text_encodings @ image_encodings.T
    )  # dist_matrix[i] gives logits for ith text

    # Note: this matrix is pretty big (5000 x 25000 with dtype float16 = 250MB)
    #  torch.argsort runs out of memory for me (6GB VRAM) so I move to CPU for sorting
    dist_matrix = dist_matrix.cpu()

    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(device)

    text_to_image_recall = []

    for k in k_vals:
        # Extract top k indices only
        topk = inds[:, :k]

        # Correct iff one of the top_k values equals the correct image (as given by text_to_image_map)
        correct = torch.eq(topk, text_to_image_map.unsqueeze(-1)).any(dim=1)

        num_correct = correct.sum().item()
        text_to_image_recall.append(num_correct / num_text)

    # image-to-text recall
    print("Image-to-text recall...")
    dist_matrix = dist_matrix.T  # dist_matrix[i] gives logits for the ith image

    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(device)

    image_to_text_recall = []

    for k in k_vals:
        # Extract top k indices only
        topk = inds[:, :k]

        correct = torch.zeros((num_im,), dtype=torch.bool).cuda()

        #  For each image, check whether one of the 5 relevant captions was retrieved
        # Check if image matches its ith caption (for i=0..4)
        for i in range(captions_per_image):
            contains_index = torch.eq(topk, image_to_text_map[:, i].unsqueeze(-1)).any(
                dim=1
            )
            correct = torch.logical_or(correct, contains_index)

        num_correct = correct.sum().item()
        image_to_text_recall.append(num_correct / num_im)  #

    print("Done.")
    return text_to_image_recall, image_to_text_recall





def coco_eval(
    model: nn.Module,
    bs: int = 1024,
    coco_root: str = "datasets",
    coco_ann_file: str = "datasets/coco/2017/annotations/captions_val2017.json",
    k_vals: List[int] = [1, 5, 10],
    text_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    vision_model_name: str = "facebook/dinov2-base",
    save_dir: str = None,
):
    model.eval()
    device = get_model_device(model)
    processor = Processor(model.vision_model.image_processor)
    dataset = CocoCaptions(
        root=coco_root,
        annFile=coco_ann_file,
        transform=processor,
        # Note: almost all images have 5 captions, but 12/5000 have 6, and 1/5000 has 7 - I ignore these few extra captions.
    )
    with autocast():
        with torch.no_grad():
            t2i, i2t = recall_at_k(
                model,
                dataset,
                device,
                k_vals=k_vals,
                batch_size=bs,
                text_model_name=text_model_name,
                vision_model_name=vision_model_name,
                save_dir=save_dir,
            )
    result_dict = {}
    print("Text-to-image Recall@K")
    for k, x in zip(k_vals, t2i):
        print(f" R@{k}: {100*x:.2f}%")
        result_dict[f"T2I R@{k}"] = x

    print("Image-to-text Recall@K")
    for k, x in zip(k_vals, i2t):
        print(f" R@{k}: {100*x:.2f}%")
        result_dict[f"I2T R@{k}"] = x

    return result_dict
    
