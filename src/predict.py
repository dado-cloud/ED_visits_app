
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet


def predict_daily_forecast(model, training_dataset, future_df):
    # Create dataset for prediction
    predict_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        future_df,
        predict=True,
        stop_randomization=True
    )

    # Convert to dataloader
    dataloader = predict_dataset.to_dataloader(
        train=False,
        batch_size=64,
        num_workers=0
    )

    # Run prediction
    predictions = model.predict(dataloader)

    # Take median quantile (middle)
    preds = predictions.numpy()
    if preds.ndim == 3:
        preds = preds[:, :, preds.shape[2] // 2]

    preds = preds.flatten()

    # Extract corresponding timestamps
    dates = future_df["date"].tail(len(preds)).values

    result_df = pd.DataFrame({
        "date": dates,
        "predicted_visits": preds
    })

    return result_df
