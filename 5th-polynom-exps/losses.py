import torch 


class Losses():
    """Loss functions for training"""

    @staticmethod
    def ce():
        return torch.nn.CrossEntropyLoss()

    @staticmethod
    def weighted_ce(weights: list[float]):
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        return torch.nn.CrossEntropyLoss(weight=weights_tensor)

    @staticmethod
    def label_smooth(epsilon: float = 0.1):
        return torch.nn.CrossEntropyLoss(label_smoothing=epsilon)


