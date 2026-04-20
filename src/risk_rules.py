import pandas as pd


def calculate_risk_level(avg_visits):
    """
    Assign a simple risk level based on average predicted visits.
    Adjust these thresholds later based on your hospital baseline.
    """
    if avg_visits < 20:
        return "Low"
    elif avg_visits < 30:
        return "Moderate"
    else:
        return "High"


def generate_alerts(df):
    """
    Generate alert messages for unusually high predicted demand.
    """
    alerts = []

    if df.empty:
        return alerts

    mean_visits = df["predicted_visits"].mean()
    threshold = mean_visits * 1.2

    high_days = df[df["predicted_visits"] > threshold]

    for _, row in high_days.iterrows():
        alerts.append(
            f"Predicted demand is high on {pd.to_datetime(row['date']).date()} "
            f"with {row['predicted_visits']:.0f} expected visits."
        )

    return alerts


def get_peak_periods_table(df, top_n=5):
    """
    Return the top forecasted periods with a simple risk label.
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "predicted_visits", "risk"])

    peak_df = df.copy()
    mean_visits = peak_df["predicted_visits"].mean()

    def label_risk(x):
        if x < mean_visits:
            return "Normal"
        elif x < mean_visits * 1.2:
            return "Elevated"
        else:
            return "High"

    peak_df["risk"] = peak_df["predicted_visits"].apply(label_risk)
    peak_df = peak_df.sort_values("predicted_visits", ascending=False).head(top_n)

    peak_df["date"] = pd.to_datetime(peak_df["date"]).dt.date

    return peak_df[["date", "predicted_visits", "risk"]]
