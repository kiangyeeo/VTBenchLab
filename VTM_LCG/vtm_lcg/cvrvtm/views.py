from __future__ import annotations

from collections.abc import Mapping

from PIL import Image
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class DeterministicPairedViewTransform:
    """Create an aligned clean view and a deterministic flipped/photometric view."""

    def __init__(
        self,
        preprocess_config: Mapping[str, object],
        view_config: Mapping[str, object],
    ) -> None:
        interpolation_name = str(
            preprocess_config.get("interpolation", "bicubic")
        ).upper()
        try:
            interpolation = InterpolationMode[interpolation_name]
        except KeyError as error:
            raise ValueError(
                f"Unsupported interpolation: {interpolation_name}"
            ) from error
        resize_size = int(preprocess_config["resize_size"])
        input_size = int(preprocess_config["input_size"])
        self.geometry = transforms.Compose(
            [
                transforms.Resize(
                    resize_size,
                    interpolation=interpolation,
                    antialias=True,
                ),
                transforms.CenterCrop(input_size),
            ]
        )
        self.mean = tuple(float(value) for value in preprocess_config["mean"])
        self.std = tuple(float(value) for value in preprocess_config["std"])
        self.horizontal_flip = bool(view_config.get("horizontal_flip", True))
        self.brightness_delta = float(view_config.get("brightness_delta", 0.1))
        self.contrast_delta = float(view_config.get("contrast_delta", 0.1))
        self.saturation_delta = float(view_config.get("saturation_delta", 0.1))
        for name, value in (
            ("brightness_delta", self.brightness_delta),
            ("contrast_delta", self.contrast_delta),
            ("saturation_delta", self.saturation_delta),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0,1)")

    def _to_normalized_tensor(self, image: Image.Image) -> Tensor:
        tensor = TF.to_tensor(image)
        return TF.normalize(tensor, self.mean, self.std)

    def __call__(self, image: Image.Image, image_id: int) -> tuple[Tensor, Tensor]:
        base = self.geometry(image)
        view_a = self._to_normalized_tensor(base)
        view_b = TF.hflip(base) if self.horizontal_flip else base.copy()
        sign = 1.0 if (int(image_id) * 2_654_435_761) % 2 == 0 else -1.0
        view_b = TF.adjust_brightness(
            view_b,
            1.0 + sign * self.brightness_delta,
        )
        view_b = TF.adjust_contrast(
            view_b,
            1.0 - sign * self.contrast_delta,
        )
        view_b = TF.adjust_saturation(
            view_b,
            1.0 + sign * self.saturation_delta,
        )
        return view_a, self._to_normalized_tensor(view_b)


def align_flipped_patch_tokens(
    values: Tensor,
    *,
    grid_shape: tuple[int, int],
    horizontal_flip: bool,
) -> Tensor:
    if not horizontal_flip:
        return values
    rows, columns = grid_shape
    if values.ndim != 3 or values.shape[1] != rows * columns:
        raise ValueError(
            f"Expected patch tokens [B,{rows * columns},D], got {tuple(values.shape)}"
        )
    return values.reshape(values.shape[0], rows, columns, values.shape[-1]).flip(2).reshape(
        values.shape
    )
