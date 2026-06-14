from mmengine.config import Config
from mmengine.runner import Runner
import torch
import torch.nn as nn
import sys
import os
import random
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from mmseg.models.segmentors import BaseSegmentor

# from open_clip import tokenizer
from model.sail_model import SAILModel
from model import get_cast_dtype


from .imagenet_template import *
from .custom_datasets import *

@MODELS.register_module()
class SAILModelSegmentation(BaseSegmentor):
    def __init__(
        self,
        head_weights_path,
        text_model_name,
        vision_model_name,
        target_dimension,
        linear_type,
        device,
        precision: str = "fp32",
        name_path="./cls_ade20k.txt",
        prob_thd=0.0,
        slide_stride=224,
        slide_crop=448,
        save_dir: str = "./evaluation/backbone_features",
        agg_mode='concat' 
    ):
        # USED for CLIP model, where scale is multipled by 255 to match the scale of 0-255
        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            bgr_to_rgb=True,
        )
        
        # below are used for DINO models
        # data_preprocessor = SegDataPreProcessor(
        #     mean=[123.675,116.28,103.53],
        #     std=[58.395,57.12,57.375],
        #     bgr_to_rgb=True,
        # )

        super().__init__(data_preprocessor=data_preprocessor)
        cast_dtype = get_cast_dtype(precision)
        self.net = SAILModel(
            text_model_name=text_model_name,
            vision_model_name=vision_model_name,
            target_dimension=target_dimension,
            vlhead_weights_path=head_weights_path,
            linear_type=linear_type,
            cast_dtype=cast_dtype,
            seg=True,
            agg_mode=agg_mode
        )

        self.net.eval().to(device)
        self.tokenizer = self.net.text_model.tokenizer
        self.patch_size = (
            self.net.vision_model.model.config.patch_size,
            self.net.vision_model.model.config.patch_size,
        )

        query_words, self.query_idx = get_cls_idx(name_path)
        self.task = name_path.split("/")[-1].split(".")[0]
        self.num_queries = len(query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)
        self.save_backbone_features_path = os.path.join(
            save_dir, f"{vision_model_name}/seg_{self.task}.pt"
        )
        self.save_backbone_query_features_path = os.path.join(
            save_dir, f"{text_model_name}/seg_{self.task}.pt"
        )
        os.makedirs(os.path.join(save_dir, text_model_name), exist_ok=True)
        os.makedirs(os.path.join(save_dir, vision_model_name), exist_ok=True)
        query_features = []
        if not os.path.exists(self.save_backbone_query_features_path):
            print("Encoding query features")
            pre_encode_model_features = {}
            with torch.no_grad():  # sub_imagenet_template, openai_imagenet_template
                for qw in query_words:
                    query = self.tokenizer(
                        [temp(qw) for temp in openai_imagenet_template],
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    ).to(device)
                    feature, encode_features = self.net.encode_text(
                        query, text_list=[temp(qw) for temp in openai_imagenet_template], return_encoded=True
                    )
                    pre_encode_model_features[qw] = encode_features.cpu()
                    feature /= feature.norm(dim=-1, keepdim=True)
                    feature = feature.mean(dim=0)
                    feature /= feature.norm()
                    query_features.append(feature.unsqueeze(0))
            torch.save(
                pre_encode_model_features, self.save_backbone_query_features_path
            )
        else:
            self.query_features = torch.load(self.save_backbone_query_features_path)
            with torch.no_grad():
                for qw in query_words:
                    encoded_features = self.query_features[qw].to(device)
                    feature = self.net.encode_text(
                        encoded_features, is_pre_encoded=True
                    )
                    feature /= feature.norm(dim=-1, keepdim=True)
                    feature = feature.mean(dim=0)
                    feature /= feature.norm()
                    query_features.append(feature.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0)

        if not os.path.exists(self.save_backbone_features_path):
            self.img_encode_features = {}
        else:
            self.img_encode_features = torch.load(self.save_backbone_features_path)
        self.dtype = self.query_features.dtype
        self.logit_scale = self.net.vlhead.logit_scale
        self.prob_thd = prob_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop

    def forward_feature(self, img, logit_size=None, attetion_type='qk', ignore_residual=False):
        if type(img) == list:
            img = img[0]

        patch_features, _ = self.net.encode_image(
            {"pixel_values": img}, return_encoded=True, patch_mode=True, attetion_type=attetion_type, ignore_residual=ignore_residual,
        )
        # if img_name not in self.img_encode_features:
        #     image_features, encoded_features = self.net.encode_image_patch(
        #         {"pixel_values": img}, return_encoded=True
        #     )
        #     self.img_encode_features[img_name] = encoded_features.cpu()
        # else:
        #     encoded_features = self.img_encode_features[img_name].to(img.device)
        #     image_features = self.net.encode_image_patch(
        #         encoded_features, is_pre_encoded=True
        #     )
        # torch.Size([1, 784, 512]), 512 is the dimension of the image features
        patch_features /= patch_features.norm(dim=-1, keepdim=True)
        # torch.Size([1, 784, 150])
        logits = patch_features @ self.query_features.T
        
        w, h = (
            img[0].shape[-2] // self.patch_size[0],
            img[0].shape[-1] // self.patch_size[1],
        )
        out_dim = logits.shape[-1]
        logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h)
        if logit_size == None:
            logits = nn.functional.interpolate(
                logits, size=img.shape[-2:], mode="bilinear"
            )
        else:
            logits = nn.functional.interpolate(logits, size=logit_size, mode="bilinear")

        return logits

    def forward_slide(self, img, img_metas, stride=112, crop_size=224, attetion_type='qk', ignore_residual=False):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)
        img_name = img_metas[0]["img_path"].split("/")[-1]
        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = self.num_queries
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]

                # pad image when (image_size % patch_size != 0)
                H, W = crop_img.shape[2:]
                pad = self.compute_padsize(H, W, self.patch_size[0])

                if any(pad):
                    crop_img = nn.functional.pad(crop_img, pad)

                crop_seg_logit = self.forward_feature(crop_img, attetion_type=attetion_type, ignore_residual=ignore_residual)

                # mask cutting for padded image
                if any(pad):
                    l, t = pad[0], pad[2]
                    crop_seg_logit = crop_seg_logit[:, :, t : t + H, l : l + W]
                preds += nn.functional.pad(
                    crop_seg_logit,
                    (
                        int(x1),
                        int(preds.shape[3] - x2),
                        int(y1),
                        int(preds.shape[2] - y2),
                    ),
                )

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat
        img_size = img_metas[0]["ori_shape"][:2]
        logits = nn.functional.interpolate(preds, size=img_size, mode="bilinear")

        return logits

    @torch.no_grad()
    def predict(self, inputs, data_samples, attetion_type='qk', ignore_residual=False):
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0],
                    img_path="tmp.png",
                )
            ] * inputs.shape[0]
        inputs = inputs.half()
        if self.slide_crop > 0:
            seg_logits = self.forward_slide(
                inputs, batch_img_metas, self.slide_stride, self.slide_crop, attetion_type, ignore_residual
            )
        else:
            seg_logits = self.forward_feature(inputs, batch_img_metas[0]["ori_shape"], attetion_type=attetion_type, ignore_residual=ignore_residual)

        return self.postprocess_result(seg_logits, data_samples)

    def postprocess_result(self, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            seg_logits = seg_logits[i] * self.logit_scale
            seg_logits = seg_logits.softmax(0)  # n_queries * w * h

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]

            seg_pred = seg_logits.argmax(0, keepdim=True)
            seg_pred[seg_logits.max(0, keepdim=True)[0] < self.prob_thd] = 0

            if data_samples is None:
                return seg_pred
            else:
                data_samples[i].set_data(
                    {
                        "seg_logits": PixelData(**{"data": seg_logits}),
                        "pred_sem_seg": PixelData(**{"data": seg_pred}),
                    }
                )
        return data_samples

    def compute_padsize(self, H: int, W: int, patch_size: int):
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l

        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t

        return l, r, t, b

    def _forward(data_samples):
        """ """

    def inference(self, img, batch_img_metas):
        """ """

    def encode_decode(self, inputs, batch_img_metas):
        """ """

    def extract_feat(self, inputs):
        """ """

    def loss(self, inputs, data_samples):
        """ """

    def save_backbone_features(self):
        if not os.path.exists(self.save_backbone_features_path):
            torch.save(self.img_encode_features, self.save_backbone_features_path)
            print(f"Save backbone features to {self.save_backbone_features_path}")


def get_cls_idx(path):
    with open(path, "r") as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(",")
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace("\n", "") for item in class_names]
    return class_names, class_indices


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if "visualization" in default_hooks:
        visualization_hook = default_hooks["visualization"]
        # Turn on visualization
        visualization_hook["draw"] = True
        if args.show:
            visualization_hook["show"] = True
            visualization_hook["wait_time"] = args.wait_time
        if args.show_dir:
            visualizer = cfg.visualizer
            visualizer["save_dir"] = args.show_dir
    else:
        raise RuntimeError(
            "VisualizationHook must be included in default_hooks."
            "refer to usage "
            "\"visualization=dict(type='VisualizationHook')\""
        )

    return cfg


import matplotlib.pyplot as plt
import numpy as np


def visualize_segmentation(img_tensor, seg_pred, save_path=None):
    """
    Visualize the segmentation result by showing the original image and predicted segmentation.

    Args:
        img_tensor (torch.Tensor): The input image tensor of shape (C, H, W).
        seg_pred (torch.Tensor): The predicted segmentation mask of shape (H, W).
        save_path (str): If provided, the plot will be saved at this path.
    """
    # Convert img_tensor from Tensor to NumPy and transpose to (H, W, C)
    img = img_tensor.transpose(1, 2, 0)

    # Convert segmentation prediction to NumPy
    seg_mask = seg_pred

    # Create a plot with two subplots: the original image and the segmentation result
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot original image
    ax[0].imshow(img.astype(np.uint8))
    ax[0].axis("off")
    ax[0].set_title("Original Image")

    # Plot segmentation mask with a color map
    ax[1].imshow(seg_mask, cmap="viridis")
    ax[1].axis("off")
    ax[1].set_title("Segmentation Result")

    # Adjust layout and save or show the result
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Segmentation result saved to {save_path}")
    else:
        plt.show()


# Example usage inside your script after getting the segmentation prediction:
# visualize_segmentation(img_tensor, seg_pred, save_path='images/seg_pred.png')


def segmentation_eval(
    text_model_name,
    vision_model_name,
    head_weights_path,
    linear_type,
    target_dimension,
    device,
    task_config,
    save_dir,
    precision="fp32",
    visualize=False,
):

    cfg = Config.fromfile(task_config)
    cfg.launcher = "none"
    cfg.work_dir = "./segmentation"
    cfg.model.text_model_name = text_model_name
    cfg.model.vision_model_name = vision_model_name
    cfg.model.head_weights_path = head_weights_path
    cfg.model.linear_type = linear_type
    cfg.model.target_dimension = target_dimension
    cfg.model.device = device
    cfg.model.save_dir = save_dir
    cfg.model.precision = precision
    # trigger_visualization_hook(cfg, args)
    runner = Runner.from_cfg(cfg)
    runner.model.to(device)
    
    if visualize == True:
        random_index = random.randint(0, len(runner.test_dataloader.dataset))
        random_index = 1
        img_tensor = runner.test_dataloader.dataset[random_index]["inputs"]
        img_tensor = torch.tensor(img_tensor).to("cuda").unsqueeze(0)
        seg_pred = runner.model.predict(
            img_tensor,
            data_samples=[runner.test_dataloader.dataset[random_index]["data_samples"]],
        )
        seg_pred = seg_pred[0].pred_sem_seg.data.cpu().numpy().squeeze(0)
        visualize_segmentation(
            img_tensor.cpu().numpy().squeeze(0),
            seg_pred,
            save_path="seg_pred_qq_res.png",
        )
    else:
        results = runner.test()
        # runner.model.save_backbone_features()

        return results