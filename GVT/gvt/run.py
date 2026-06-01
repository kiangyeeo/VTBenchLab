import os
import copy
import torch
from pathlib import Path
from collections import OrderedDict
import pytorch_lightning as pl

from gvt.config import ex
from gvt.modules import GVT
from gvt.datamodules.multitask_datamodule import MTDataModule

from pytorch_lightning.plugins import environments as pl_env
from pytorch_lightning.utilities.distributed import rank_zero_info

import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def normalize_checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    normalized = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]
        normalized[new_key] = value

    # The refactor wraps the original visual_encoder under visual_tokenizer.
    # Keep official checkpoints compatible by providing both key namespaces.
    for key, value in list(normalized.items()):
        if key.startswith("visual_encoder."):
            wrapped_key = "visual_tokenizer." + key
            normalized.setdefault(wrapped_key, value)

    return normalized


def load_checkpoint_for_eval(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = normalize_checkpoint_state_dict(checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)

    model_keys = set(model.state_dict().keys())
    loaded_keys = sorted(key for key in state_dict.keys() if key in model_keys)
    important_prefixes = (
        "visual_encoder.",
        "visual_tokenizer.",
        "ln_vision.",
        "llm_proj.",
        "input_ln.",
        "perceiver.",
        "token_dim_proj.",
        "vision_adapter.",
    )
    loaded_important = [key for key in loaded_keys if key.startswith(important_prefixes)]
    missing_important = [
        key for key in incompatible.missing_keys
        if key.startswith(important_prefixes)
    ]
    unexpected_important = [
        key for key in incompatible.unexpected_keys
        if key.startswith(important_prefixes)
    ]

    print("load state dict from:", checkpoint_path)
    print(
        "checkpoint load summary:",
        f"loaded={len(loaded_keys)}",
        f"loaded_important={len(loaded_important)}",
        f"missing_important={len(missing_important)}",
        f"unexpected_important={len(unexpected_important)}",
    )
    if missing_important:
        print("missing important keys sample:", missing_important[:20])
    if unexpected_important:
        print("unexpected important keys sample:", unexpected_important[:20])

    if len(loaded_important) == 0:
        raise RuntimeError(
            "No GVT bridge/vision checkpoint tensors were loaded. "
            "Check that load_path points to the official GVT checkpoint."
        )


class OMPIClusterEnvironment(pl_env.ClusterEnvironment):
    def __init__(self):
        super().__init__()

    @property
    def creates_processes_externally(self):
        return True

    def world_size(self) -> int:
        return int(os.environ["OMPI_COMM_WORLD_SIZE"])

    def set_world_size(self, size: int):
        pass

    def global_rank(self) -> int:
        return int(os.environ["OMPI_COMM_WORLD_RANK"])

    def set_global_rank(self, rank: int):
        pass

    def local_rank(self) -> int:
        return int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])

    def node_rank(self) -> int:
        if "NODE_RANK" in os.environ:
            return int(os.environ["NODE_RANK"])
        else:
            return 0

    def master_address(self) -> str:
        return os.environ["MASTER_ADDR"]

    def master_port(self) -> int:
        return int(os.environ["MASTER_PORT"])


def get_cluster_plugin(num_gpus=1, num_nodes=1):
    if num_nodes > 1 or (
        num_nodes == 1 and "OMPI_COMM_WORLD_SIZE" in os.environ
    ):
        rank_zero_info("ClusterPlugin: using OMPI Cluster Environment")
        return OMPIClusterEnvironment()
    if num_gpus >= 1:
        rank_zero_info("ClusterPlugin: using Lightning Cluster Environment")
        return pl_env.LightningEnvironment()
    return None



@ex.automain
def main(_config):

    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    output_dir = Path(_config.get("output_dir") or ".")
    pred_result_dir = Path(_config.get("pred_result_dir") or output_dir / "pred_results")
    log_dir = Path(_config.get("log_dir") or output_dir / "output")
    eval_gt_dir = Path(_config["data_root"]) / "eval_gt" if _config.get("data_root") else output_dir / "eval_gt"

    pred_result_dir.mkdir(parents=True, exist_ok=True)
    (pred_result_dir / "count").mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    eval_gt_dir.mkdir(parents=True, exist_ok=True)

    _config["pred_result_dir"] = str(pred_result_dir)
    _config["log_dir"] = str(log_dir)

    use_distributed_sampler = _config.get("num_gpus", 1) * _config.get("num_nodes", 1) > 1
    dm = MTDataModule(_config, dist=use_distributed_sampler)

    baseline = _config['use_baseline']
    if baseline:
        if baseline == "llava":
            from gvt.modules import MLLM_LLAVA
            model = MLLM_LLAVA(config=_config)
        elif baseline == "minigpt4":
            from gvt.modules import MLLM_MINIGPT4
            resume_ckpt = "params/pretrained_minigpt4_7b.pth"
            model = MLLM_MINIGPT4(config=_config)
        elif baseline == "blip2":
            from gvt.modules import MLLM_BLIP2
            model = MLLM_BLIP2(config=_config)
        else:
            raise ValueError(f"Unknown use_baseline='{baseline}'. Expected llava, minigpt4, blip2, or empty.")
    else:
        model = GVT(config=_config)
    
    exp_name = f'{_config["exp_name"]}'
    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=-1,
        verbose=True,
        monitor="val/the_metric",
        mode="min",
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}',
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    resume_ckpt = _config['load_path']
    if _config["test_only"] and resume_ckpt is not None:
        load_checkpoint_for_eval(model, resume_ckpt)

    trainer_kwargs = dict(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="gpu",
        benchmark=True,
        deterministic=False,
        max_epochs=_config["max_epoch"] if max_steps is None else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        replace_sampler_ddp=False,
        accumulate_grad_batches=1,
        log_every_n_steps=10,
        resume_from_checkpoint=None if _config["test_only"] else resume_ckpt,
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
    )
    if _config.get("use_deepspeed", False):
        trainer_kwargs["strategy"] = "deepspeed_stage_2"

    trainer = pl.Trainer(**trainer_kwargs)

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        model.eval()
        with torch.no_grad():
            trainer.test(model, datamodule=dm)
