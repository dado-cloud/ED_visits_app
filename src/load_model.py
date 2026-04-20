import torch
from pytorch_forecasting import TemporalFusionTransformer


def load_tft_model(model_path):
    model = TemporalFusionTransformer.load_from_checkpoint(
        model_path,
        map_location=torch.device("cpu")
    )
    model.eval()
    return model


def load_training_dataset(path):
    training = torch.load(
        path,
        map_location=torch.device("cpu"),
        weights_only=False
    )
    return training
