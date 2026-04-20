import cloudpickle
import torch
from pytorch_forecasting import TemporalFusionTransformer


def load_tft_model(model_path):
    model = TemporalFusionTransformer.load_from_checkpoint(
        model_path,
        map_location=torch.device("cpu")
    )
    model.eval()
    torch.set_grad_enabled(False)
    return model




def load_training_dataset(path):
    with open(path, "rb") as f:
        training = cloudpickle.load(f)
    return training
