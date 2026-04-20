import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet


def predict_daily_forecast(model, training_dataset, future_df):
    # build prediction dataset from the saved training dataset structure
    predict_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        future_df,
        predict=True,
        stop_randomization=True
    )

    # dataloader
    dataloader = predict_dataset.to_dataloader(
        train=False,
        batch_size=64,
        num_workers=0
    )

    # predict
    predictions = model.predict(dataloader)

    # convert to numpy
    if hasattr(predictions, "cpu"):
        preds = predictions.cpu().numpy()
    else:
        preds = predictions.numpy()

    # if quantile output exists, take middle quantile (median)
    if preds.ndim == 3:
        preds = preds[:, :, preds.shape[2] // 2]

    preds = preds.flatten()

    # reverse log1p transformation used during training
    preds = np.expm1(preds)

    # avoid negative outputs after inverse transform
    preds = np.maximum(preds, 0)

    # get only future dates
    future_dates = future_df[future_df["ED_visits"] == 0]["date"].reset_index(drop=True)

    # align lengths safely
    n = min(len(future_dates), len(preds))
    result_df = pd.DataFrame({
        "date": future_dates.iloc[:n],
        "predicted_visits": preds[:n]
    })

    return result_df
