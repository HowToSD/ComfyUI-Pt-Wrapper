from typing import Any, Dict, Tuple
import torchvision.transforms as transforms


class PtvTransformsDataAugment:
    """
    Applies data augmentation transformations to dataset elements.
    Supported transformations include flipping, rotation, translation, and scaling.

    category: PyTorch wrapper - Training
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Defines the required input types for the transformations.

        Returns:
            Dict[str, Any]: A dictionary specifying the required input types and their constraints.
        """
        return {
            "required": {
                "h_flip_prob": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "v_flip_prob": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "rotate_degree": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.001}),
                "h_translate_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "v_translate_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "min_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.0, "step": 0.001}),
                "max_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.001})
            },
        }

    RETURN_TYPES: Tuple[str] = ("PTVTRANSFORM",)
    FUNCTION: str = "f"
    CATEGORY: str = "Training"

    def f(self,
          h_flip_prob: float,
          v_flip_prob: float,
          rotate_degree: float,
          h_translate_ratio: float,
          v_translate_ratio: float,
          min_scale: float,
          max_scale: float) -> Tuple[transforms.Compose]:
        """
        Applies the specified transformations to the dataset.

        Args:
            h_flip_prob (float): Probability of applying horizontal flip.
            v_flip_prob (float): Probability of applying vertical flip.
            rotate_degree (float): Maximum rotation degree.
            h_translate_ratio (float): Horizontal translation ratio.
            v_translate_ratio (float): Vertical translation ratio.
            min_scale (float): Minimum scale factor.
            max_scale (float): Maximum scale factor.

        Returns:
            Tuple[transforms.Compose]: A tuple containing the transformation pipeline.
        """
        translate_tuple = (h_translate_ratio, v_translate_ratio) if h_translate_ratio > 0.0 or v_translate_ratio > 0.0 else None
        scale_tuple = (min_scale, max_scale) if min_scale != 1.0 or max_scale != 1.0 else None

        actions = []
        if h_flip_prob > 0.0:
            actions.append(transforms.RandomHorizontalFlip(p=h_flip_prob))
        if v_flip_prob > 0.0:
            actions.append(transforms.RandomVerticalFlip(p=v_flip_prob))

        actions.append(
            transforms.RandomAffine(
                degrees=rotate_degree,
                translate=translate_tuple,
                scale=scale_tuple
            )
        )
        actions.append(transforms.ToTensor())  # Converts image to tensor and scales to [0, 1]

        transform_pipeline = transforms.Compose(actions)

        return (transform_pipeline,)
