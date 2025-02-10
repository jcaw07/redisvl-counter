import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# For the linear regression
from sklearn.linear_model import LinearRegression

plt.rcParams.update({"figure.figsize": (10, 6), "font.size": 12})

def load_data(csv_path):
    """
    Loads daily download data from a CSV into a pandas DataFrame.
    Expects columns: [date, downloads].
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    return df

def filter_by_date_range(df, start_date, end_date):
    """
    Returns rows from df where date is between start_date and end_date (inclusive).
    start_date, end_date can be strings or datetime objects.
    """
    start_dt = pd.to_datetime(start_date)
    end_dt   = pd.to_datetime(end_date)
    mask = (df["date"] >= start_dt) & (df["date"] <= end_dt)
    return df[mask]

def compute_quarter_metrics(df, quarter_label, start_date, end_date):
    """
    Filter df to a quarter date range, compute metrics:
      - total downloads
      - average daily
    """
    sub_df = filter_by_date_range(df, start_date, end_date)
    
    total_downloads = sub_df["downloads"].sum()
    num_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1  # inclusive
    avg_daily = total_downloads / num_days if num_days > 0 else 0

    metrics = {
        "quarter": quarter_label,
        "start_date": start_date,
        "end_date": end_date,
        "total_downloads": total_downloads,
        "avg_daily_downloads": avg_daily
    }
    return metrics, sub_df

def compute_monthly_metrics(sub_df):
    """
    Given a DataFrame of daily data, return monthly totals and average daily per month.
    We'll group by year-month (YYYY-MM).
    """
    # Extract year-month from the 'date'
    # .dt.to_period("M") yields a Period (e.g., 2024-08)
    sub_df["year_month"] = sub_df["date"].dt.to_period("M")
    
    # Group by year_month, sum downloads
    monthly_group = sub_df.groupby("year_month")["downloads"].sum().reset_index()
    monthly_group.rename(columns={"downloads": "monthly_downloads"}, inplace=True)
    
    # Count how many daily rows exist for each year_month
    daily_counts = sub_df.groupby("year_month")["date"].count().reset_index()
    daily_counts.rename(columns={"date": "days_in_month_sample"}, inplace=True)
    
    # Merge them
    monthly_stats = pd.merge(monthly_group, daily_counts, on="year_month")
    
    # Compute average daily
    monthly_stats["avg_daily_downloads"] = (
        monthly_stats["monthly_downloads"] / monthly_stats["days_in_month_sample"]
    )
    
    # Convert year_month back to string to make it easier to plot
    monthly_stats["year_month"] = monthly_stats["year_month"].astype(str)
    
    return monthly_stats

def fit_linear_forecast(df, forecast_days=180):
    """
    Fits a simple linear regression on df (which should have 'date' and 'downloads').
    Returns:
      - A new DataFrame with columns ['date', 'predicted_downloads']
        covering the entire historical range + the next `forecast_days`.
      - The model object if you want additional stats.
    
    Approach:
      - Convert date to an integer day offset (X) from the earliest date.
      - Fit a linear regression to (X, downloads).
      - Extrapolate X for the next forecast_days beyond the last date in df.
    """
    if df.empty:
        return pd.DataFrame(), None

    df = df.copy()
    df.sort_values("date", inplace=True)

    # 1) Convert date to numeric day offset
    min_date = df["date"].min()
    df["day_offset"] = (df["date"] - min_date).dt.days
    
    X = df[["day_offset"]]            # shape (n_samples, 1)
    y = df["downloads"].values        # shape (n_samples,)
    
    # 2) Fit linear regression
    model = LinearRegression()
    model.fit(X, y)

    # 3) Create a date range from earliest date to last date + forecast_days
    last_date = df["date"].max()
    forecast_range = pd.date_range(
        start=min_date,
        end=last_date + pd.Timedelta(days=forecast_days),
        freq="D"
    )
    
    forecast_df = pd.DataFrame({"date": forecast_range})
    forecast_df["day_offset"] = (forecast_df["date"] - min_date).dt.days
    
    # 4) Predict for the entire range
    X_forecast = forecast_df[["day_offset"]]
    y_pred = model.predict(X_forecast)
    forecast_df["predicted_downloads"] = y_pred
    
    return forecast_df, model

def main(csv_path):
    # 1) Load your daily data
    df = load_data(csv_path)
    if df.empty:
        print("No data found in CSV.")
        sys.exit(1)
    
    # 2) Define Q3 and Q4 date ranges (per your specs)
    #    Q3: 2024-08-01 to 2024-10-31
    #    Q4: 2024-11-01 to 2025-01-31
    quarters_info = [
        ("Q3 (Aug-Oct 2024)", "2024-08-01", "2024-10-31"),
        ("Q4 (Nov 2024-Jan 2025)", "2024-11-01", "2025-01-31"),
    ]
    
    quarter_results = []
    combined_sub_df = pd.DataFrame()
    
    # 3) Compute quarter metrics & gather data
    for (label, start_date, end_date) in quarters_info:
        q_metrics, q_df = compute_quarter_metrics(df, label, start_date, end_date)
        quarter_results.append(q_metrics)
        # Combine Q3+Q4 daily data
        combined_sub_df = pd.concat([combined_sub_df, q_df], ignore_index=True)
    
    q_df_metrics = pd.DataFrame(quarter_results)
    
    # 4) Growth from Q3 -> Q4
    if len(q_df_metrics) == 2:
        q3_downloads = q_df_metrics.loc[0, "total_downloads"]
        q4_downloads = q_df_metrics.loc[1, "total_downloads"]
        if q3_downloads > 0:
            growth_pct = (q4_downloads - q3_downloads) / q3_downloads * 100
        else:
            growth_pct = None
    else:
        growth_pct = None
    
    # 5) Monthly metrics (for Q3 & Q4 combined)
    monthly_stats = compute_monthly_metrics(combined_sub_df)

    # 6) Print the story / summary
    print("\n=== QUARTERLY SUMMARY (Q3 & Q4) ===\n")
    print(q_df_metrics[["quarter", "total_downloads", "avg_daily_downloads"]])
    print("")
    
    if growth_pct is not None:
        print(f"Growth from Q3 to Q4 (total downloads): {growth_pct:.2f}%\n")
    else:
        print("Unable to compute Q3 -> Q4 growth (missing or zero data).\n")
    
    print("=== MONTHLY BREAKDOWN (within Q3 & Q4) ===")
    print(monthly_stats[["year_month", "monthly_downloads", "avg_daily_downloads"]])
    print("")

    # ----- Charts -----

    # A) Bar chart: Quarterly totals (Q3 vs Q4)
    plt.figure()
    plt.bar(q_df_metrics["quarter"], q_df_metrics["total_downloads"], color=["#668cff", "#ff8533"])
    plt.title("Quarterly Total Downloads (Q3 vs Q4)")
    plt.ylabel("Total Downloads")
    for i, val in enumerate(q_df_metrics["total_downloads"]):
        plt.text(i, val + 0.01 * val, str(val), ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.show()
    
    # B) Bar chart: Monthly totals for Q3 & Q4
    plt.figure()
    plt.bar(monthly_stats["year_month"], monthly_stats["monthly_downloads"], color="#77dd77")
    plt.title("Monthly Total Downloads (Q3 & Q4)")
    plt.ylabel("Downloads")
    plt.xticks(rotation=45)
    for i, val in enumerate(monthly_stats["monthly_downloads"]):
        plt.text(i, val + 0.01 * val, str(val), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.show()
    
    # C) Daily line chart for entire Q3 & Q4 range + linear forecast
    #    We'll forecast 6 months (roughly 2 quarters) beyond the last Q4 date.
    #    That means beyond 2025-01-31.

    # 1. Filter daily data from Q3 start to Q4 end
    overall_start = pd.to_datetime(quarters_info[0][1])  # 2024-08-01
    overall_end   = pd.to_datetime(quarters_info[-1][2]) # 2025-01-31
    overall_df = filter_by_date_range(df, overall_start, overall_end)
    
    if overall_df.empty:
        print("No daily data for Q3 & Q4 date range. Cannot plot or forecast.")
        return
    
    # 2. Fit a linear model on the daily data from Q3 & Q4
    forecast_days = 180  # ~6 months
    forecast_df, model = fit_linear_forecast(overall_df, forecast_days=forecast_days)
    
    # 3. Separate the forecast_df into "historical" vs. "future" data
    last_obs_date = overall_df["date"].max()
    hist_mask = forecast_df["date"] <= last_obs_date
    future_mask = forecast_df["date"] > last_obs_date
    
    plt.figure()
    # Plot actual daily downloads as scatter or line
    plt.plot(overall_df["date"], overall_df["downloads"], "o-", color="teal", label="Daily Downloads (Actual)")
    
    # Plot predicted line for historical period (matches actual dates)
    plt.plot(
        forecast_df.loc[hist_mask, "date"],
        forecast_df.loc[hist_mask, "predicted_downloads"],
        color="orange",
        linestyle="-",
        label="Linear Trend (Historical)"
    )
    
    # Plot forecast line for future period
    plt.plot(
        forecast_df.loc[future_mask, "date"],
        forecast_df.loc[future_mask, "predicted_downloads"],
        color="red",
        linestyle="--",
        label="Forecast (Next 6 Months)"
    )
    
    plt.title("Daily Downloads (Q3 & Q4) + 6-Month Linear Forecast")
    plt.xlabel("Date")
    plt.ylabel("Downloads")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 4. Optional: Summarize forecast for the next 1-2 quarters
    #    For example, predicted total downloads in the next 3 months vs. next 6 months.
    future_only = forecast_df.loc[future_mask].copy()
    if not future_only.empty:
        # We'll group by month and sum predicted downloads
        future_only["year_month"] = future_only["date"].dt.to_period("M")
        future_monthly = future_only.groupby("year_month")["predicted_downloads"].sum().reset_index()
        future_monthly["predicted_downloads"] = future_monthly["predicted_downloads"].astype(int)
        
        print("=== FORECAST: Next 6 Months (Monthly Predicted Downloads) ===")
        print(future_monthly)
        print("")
        
    print("Done. Charts displayed.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python report_Q3_Q4.py <path_to_csv>")
        sys.exit(1)
    csv_arg = sys.argv[1]
    main(csv_arg)

