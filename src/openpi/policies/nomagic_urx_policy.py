import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class NomagicURXInputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("side", "left_wrist", "right_wrist")

    def __call__(self, data: dict) -> dict:
        # Get the state. We are padding from 14 to the model action dim.
        state = np.concat([data["state"]["joints"], data["state"]["gripper"].reshape(-1)])
        state = transforms.pad_to_dim(state, self.action_dim)

        def convert_image(img):
            img = np.asarray(img)
            # Convert to uint8 if using float images.
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            # Convert from [channel, height, width] to [height, width, channel].
            img = einops.rearrange(img, "c h w -> h w c")
            return img

        images = data["images"]
        images_dict = {name: convert_image(img) for name, img in images.items()}

        data["images"] = images_dict

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        if not data["images"]:
            raise ValueError("At least one camera view is expected in 'images'")

        # Extract the size of one of the images.
        any_image = next(iter(data["images"].values()))
        image_shape = any_image.shape

        image_masks = {}

        for image_name in self.EXPECTED_CAMERAS:
            if image_name in data["images"]:
                image_masks[image_name] = np.True_
            else:
                data["images"][image_name] = np.zeros(image_shape, dtype=np.uint8)
                image_masks[image_name] = np.False_

        inputs = {
            "image": data["images"],
            "image_mask": image_masks,
            "state": state,
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.concat([data["actions"]["joints"], data["actions"]["gripper"].reshape(-1, 1)], axis=1)
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class NomagicURXOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 7 dims.
        return {"actions": np.asarray(data["actions"][:, :7])}
