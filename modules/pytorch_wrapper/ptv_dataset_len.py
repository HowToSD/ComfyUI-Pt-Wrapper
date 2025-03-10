from typing import Any, Dict


class PtvDatasetLen:
    """
    A wrapper class that calls Python len on a dataset.
    
    category: PyTorch wrapper - Training
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Defines the input types for the f function.

        Returns:
            Dict[str, Any]: A dictionary specifying required input types.
        """
        return {
            "required": {
                "dataset": ("PTVDATASET", {})
            }
        }

    RETURN_TYPES: tuple = ("INT",)
    FUNCTION: str = "f"
    CATEGORY: str = "Training"

    def f(self, dataset: Any) -> tuple:
        """
        A wrapper class that calls Python len on a dataset.

        Args:
            iterable (Any): A dataset.

        Returns:
            tuple: A tuple containing the length of the dataset.
        """
        return (len(dataset),)
    