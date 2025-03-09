import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class NomagicURXInputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("side", "left_wrist", "right_wrist")

    def __call__(self, data: dict) -> dict:
        # We want all the images to be present.
        in_images = data["images"]
        if set(in_images) != set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Construct the state.
        state = np.concat([data["state"]["arm"], data["state"]["gripper"].reshape(-1)])
        state = transforms.pad_to_dim(state, self.action_dim)

        # Change the format of the images to (H, W, C) and convert to uint8.
        for name in self.EXPECTED_CAMERAS:
            data["images"][name] = _parse_image(data["images"][name])

        image_masks = {image_name: np.True_ for image_name in self.EXPECTED_CAMERAS}

        inputs = {
            "image": data["images"],
            "image_mask": image_masks,
            "state": state,
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.concat([data["actions"]["arm"], data["actions"]["gripper"].reshape(-1, 1)], axis=1)
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class NomagicURXOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 7 dims.
        return {"actions": np.asarray(data["actions"][:, :7])}
