import pandas as pd
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"


def load_input_data(path=MODEL_DIR / "test_data.csv"):
    df = pd.read_csv(path)
    return df


def create_time_features(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    # FIX: match training naming
    df["dow"] = df["dow"] = df["date"].dt.dayofweek

    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

    df["month"] = df["date"].dt.month

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def prepare_future_dataframe(input_df, forecast_days, feature_config):
    df = input_df.copy()

    # ensure required columns exist
    required_cols = ["date", "ED_visits", "avg_weather_C", "avg_precip", "avg_snow"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # date formatting and sorting
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # static id
    if "series_id" not in df.columns:
        df["series_id"] = "ED_1"

    # sequential time index
    df["time_idx"] = np.arange(len(df))

    # IMPORTANT: match notebook training transform
    # only apply if the values look untransformed
    if df["ED_visits"].min() >= 0 and df["ED_visits"].max() > 10:
        df["ED_visits"] = np.log1p(df["ED_visits"])

    # create historical time features
    df = create_time_features(df)

    # create future horizon
    last_date = df["date"].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="D"
    )

    future_df = pd.DataFrame({"date": future_dates})
    future_df["series_id"] = "ED_1"

    # continue time_idx
    start_idx = df["time_idx"].max() + 1
    future_df["time_idx"] = np.arange(start_idx, start_idx + forecast_days)

    # placeholders for unknown reals
    future_df["ED_visits"] = 0.0
    future_df["avg_weather_C"] = df["avg_weather_C"].iloc[-1]
    future_df["avg_precip"] = df["avg_precip"].iloc[-1]
    future_df["avg_snow"] = df["avg_snow"].iloc[-1]

    # future time features
    future_df = create_time_features(future_df)

    # combine history + future
    full_df = pd.concat([df, future_df], ignore_index=True)

    return full_df
