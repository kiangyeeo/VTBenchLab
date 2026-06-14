import sys

sys.path.append("/home/mila/l/le.zhang/scratch/light_align")

from .imagenetv2 import ImageNetV2Dataset
from .imagenetv1 import ImageNetWithPaths
from .imagenet_constant import IMAGENET_CLASSES, IMAGENET_TEMPLATES
import torch

from tqdm import tqdm
import os
from typing import Union, Optional
import torch.nn as nn
from .utils import get_model_device, save_features, load_features, Processor

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Processor:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, images):
        outputs = self.processor(images, return_tensors="pt")
        if isinstance(outputs, torch.Tensor):
            return outputs[0]
        else:
            return outputs['pixel_values'][0]



def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy().item())
        for k in topk
    ]


def zeroshot_classifier(
    model,
    batch_size,
    save_backbone_classifier_features_path,
    device,
    classnames,
    templates,
    tokenizer,
    model_name,
):

    zeroshot_weights = []
    backbone_path = save_backbone_classifier_features_path

    with torch.amp.autocast(device_type='cuda'):
        with torch.no_grad():
            if os.path.exists(backbone_path):
                print(f"Loading backbone features {backbone_path} for classifier")
                pre_encode_model_features = torch.load(backbone_path, weights_only=True)
                for classname in tqdm(classnames):
                    class_features = pre_encode_model_features[classname].to(device)
                    class_embeddings = model.encode_text(
                        class_features, is_pre_encoded=True
                    )
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    class_embedding = class_embeddings.mean(dim=0)
                    class_embedding /= class_embedding.norm()
                    zeroshot_weights.append(class_embedding)
            else:
                print("Extracting backbone features for classifier")
                pre_encode_model_features = {}
                for classname in tqdm(classnames):
                    texts = [template.format(classname) for template in templates]
                    tokens = tokenizer(
                        texts, padding=True, truncation=True, return_tensors="pt"
                    ).to(device)
                    class_embeddings, class_features = model.encode_text(
                            tokens, text_list = texts, return_encoded=True
                        )
                    pre_encode_model_features[classname] = class_features.cpu()
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    class_embedding = class_embeddings.mean(dim=0)
                    class_embedding /= class_embedding.norm()
                    zeroshot_weights.append(class_embedding)
                os.makedirs(os.path.dirname(backbone_path), exist_ok=True)
                torch.save(pre_encode_model_features, backbone_path)

            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)

    return zeroshot_weights


def extract_and_save_backbone_features(
    model, device, dataloader, save_path, zeroshot_weights
):
    top1, top5, n = 0.0, 0.0, 0.0
    pre_encode_image_features = {}

    with torch.amp.autocast(device_type='cuda'):
        with torch.no_grad():
            for images, target, image_name in tqdm(dataloader):
                images, target = images.to(device), target.to(device)
                image_features, encoded_features = model.encode_image(
                    {"pixel_values": images}, return_encoded=True
                )

                for j, name in enumerate(image_name):
                    pre_encode_image_features[name] = {
                        "features": encoded_features[j].cpu(),
                        "target": target[j].cpu(),
                    }
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = 100.0 * image_features @ zeroshot_weights

                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                top1 += acc1
                top5 += acc5
                n += images.size(0)

        save_features(pre_encode_image_features, save_path)

    return top1, top5, n


def evaluate_from_saved_features(
    model, device, pre_encode_image_features, batch_size, zeroshot_weights
):
    top1, top5, n = 0.0, 0.0, 0.0
    batched_pre_encode_image_features = {}

    for i, (key, value) in enumerate(pre_encode_image_features.items()):
        if i % batch_size == 0:
            batched_pre_encode_image_features[i // batch_size] = {}
        batched_pre_encode_image_features[i // batch_size][key] = value
    with torch.amp.autocast(device_type='cuda'):
        with torch.no_grad():
            for batch in tqdm(batched_pre_encode_image_features.values()):
                encoded_features, targets = [], []
                for value in batch.values():
                    encoded_features.append(value["features"])
                    targets.append(value["target"])

                encoded_features = torch.stack(encoded_features).to(device)
                targets = torch.stack(targets).to(device)

                image_features = model.encode_image(encoded_features, is_pre_encoded=True)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = 100.0 * image_features @ zeroshot_weights

                acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
                top1 += acc1
                top5 += acc5
                n += encoded_features.size(0)

    return top1, top5, n


def imagenet_eval(
    model: nn.Module,
    bs: int = 1024,
    images_dir: str = "/home/mila/l/le.zhang/scratch/datasets",
    text_model_name: str = "all-mpnet-base-v2",
    vision_model_name: str = "dinov2-base",
    save_dir: Optional[str] = "./evaluation/backbone_features",
    version: str = "v2",
):
    if version == "v1":
        save_backbone_features_path = os.path.join(
            save_dir, f"{vision_model_name}/imagenet_v1.pt"
        )
        save_backbone_classifier_features_path = os.path.join(
            save_dir, f"{text_model_name}/classifier_v1.pt"
        )
    else:
        save_backbone_features_path = os.path.join(
            save_dir, f"{vision_model_name}/imagenet.pt"
        )
        save_backbone_classifier_features_path = os.path.join(
            save_dir, f"{text_model_name}/classifier.pt")

    model.eval()
    device = get_model_device(model)
    tokenizer = model.text_model.tokenizer
    processor = Processor(model.vision_model.image_processor)

    zeroshot_weights = zeroshot_classifier(
        model,
        bs,
        save_backbone_classifier_features_path,
        device,
        IMAGENET_CLASSES,
        IMAGENET_TEMPLATES,
        tokenizer,
        text_model_name,
    )

    if not os.path.exists(save_backbone_features_path):
        print("Extracting backbone features")
        if version == "v1":
            images_dataset = ImageNetWithPaths(
                root=images_dir, split="val", transform=processor
            )
        else:
            images_dataset = ImageNetV2Dataset(
            variant="matched-frequency", transform=processor, location=images_dir
        )
        loader = torch.utils.data.DataLoader(
            images_dataset, batch_size=bs, num_workers=2
        )
        top1, top5, n = extract_and_save_backbone_features(
            model, device, loader, save_backbone_features_path, zeroshot_weights
        )
    else:
        print(f"Loading backbone image features from {save_backbone_features_path}")
        pre_encode_image_features = load_features(save_backbone_features_path)
        top1, top5, n = evaluate_from_saved_features(
            model, device, pre_encode_image_features, bs, zeroshot_weights
        )

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100
    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")

    return {"top1": top1, "top5": top5}

