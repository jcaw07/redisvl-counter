import os
import requests
import pandas as pd
from datetime import datetime, timedelta

CSV_PATH = "redisvl_downloads.csv"
PACKAGE_NAME = "redisvl"

def fetch_overall_data(package_name, with_mirrors=False):
    """
    Fetch overall (all-time) daily download data from PyPIStats for `package_name`.
    If with_mirrors is False, we only use "without_mirrors" data.
    Otherwise, use "with_mirrors".
    
    Returns a DataFrame with columns:
      date, category, downloads
    """
    url = f"https://pypistats.org/api/packages/{package_name}/overall"
    # You can optionally do: url += "?mirrors=true" if you wanted that param,
    # but PyPIStats also supports passing mirrors=false or true at the end. We'll just parse both categories from the returned data.
    
    # Add ?mirrors=false or not, depending on your preference:
    # We'll always get all data, but we can filter out the category we don't want.
    # For clarity, let's explicitly do mirrors=false to reduce data returned:
    url += "?mirrors=false"
    
    print(f"Fetching overall data from {url}")
    
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        data_json = resp.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching overall stats: {e}")
        return pd.DataFrame()
    
    # The "overall" endpoint typically returns a structure like:
    # {
    #   "data": [
    #       {"category": "with_mirrors", "date": "2020-01-01", "downloads": 50},
    #       {"category": "without_mirrors", "date": "2020-01-01", "downloads": 30},
    #       ...
    #   ],
    #   "package": "redisvl",
    #   ...
    # }
    
    if "data" not in data_json or not isinstance(data_json["data"], list):
        print("No valid 'data' array in JSON response. Full response:")
        print(data_json)
        return pd.DataFrame()
    
    df = pd.DataFrame(data_json["data"])  # This should be a list of daily records
    if df.empty:
        print("No records found in the response.")
        return df
    
    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"])
    
    # If you only want "without_mirrors" or "with_mirrors", filter here.
    desired_category = "with_mirrors" if with_mirrors else "without_mirrors"
    df = df[df["category"] == desired_category]
    
    # Now you have daily records for your chosen category. 
    # If you wanted total (with + without), you'd group by date and sum downloads.
    
    # Sort by date ascending
    df = df.sort_values("date")
    
    return df


def load_existing_data(csv_path):
    """
    Load previously saved data from CSV if it exists, else return an empty DataFrame.
    """
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, parse_dates=["date"])
    else:
        return pd.DataFrame()


def merge_and_save(new_df, csv_path):
    """
    Merge the newly fetched data (new_df) with any existing CSV data,
    deduplicate by date, and save the result back to CSV.
    
    Returns the combined DataFrame.
    """
    existing_df = load_existing_data(csv_path)
    
    if existing_df.empty:
        combined_df = new_df
    else:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Drop duplicates by date, keeping the latest row if there's a conflict
    combined_df.drop_duplicates(subset=["date"], keep="last", inplace=True)
    
    # Sort by date again
    combined_df.sort_values("date", inplace=True)
    
    # Save to CSV
    combined_df.to_csv(csv_path, index=False)
    
    return combined_df


def compute_metrics(df):
    """
    Given a DataFrame of daily downloads (with columns: date, downloads),
    compute some sample metrics for the last year:
      - total downloads in last 365 days
      - month-over-month growth (very simple version)
    """
    # Filter to last year
    one_year_ago = pd.Timestamp.today().normalize() - pd.Timedelta(days=365)
    df_last_year = df[df["date"] >= one_year_ago]
    
    # Total downloads in the last 365 days
    total_last_year = df_last_year["downloads"].sum()
    print(f"Total downloads in the last 365 days: {total_last_year}")
    
    # Month-over-month: compare last 30 days to the 30 days before that
    if not df_last_year.empty:
        last_date = df_last_year["date"].max()
        start_period_1 = last_date - pd.Timedelta(days=29)  # last 30 days inclusive
        end_period_1 = last_date
        start_period_2 = start_period_1 - pd.Timedelta(days=30)
        end_period_2 = start_period_1 - pd.Timedelta(days=1)

        df_period_1 = df_last_year[(df_last_year["date"] >= start_period_1) & (df_last_year["date"] <= end_period_1)]
        df_period_2 = df_last_year[(df_last_year["date"] >= start_period_2) & (df_last_year["date"] <= end_period_2)]
        
        downloads_1 = df_period_1["downloads"].sum()
        downloads_2 = df_period_2["downloads"].sum()
        
        if downloads_2 > 0:
            mom_growth = (downloads_1 - downloads_2) / downloads_2 * 100
            print(f"Month-over-month growth (last 30 days vs. previous 30 days): {mom_growth:.2f}%")
        else:
            print("Month-over-month growth not available (previous 30 days had zero or missing downloads).")
    else:
        print("No downloads in the last year; can't compute MoM growth.")


def main():
    print(f"== Fetching and updating local data for package: {PACKAGE_NAME} ==")
    
    # 1) Fetch fresh overall data from PyPIStats
    new_df = fetch_overall_data(PACKAGE_NAME, with_mirrors=False)
    
    if new_df.empty:
        print("No new data fetched. Exiting.")
        return
    
    # new_df should have columns: date, category, downloads
    
    # 2) Merge with existing local CSV (if any), then save
    combined_df = merge_and_save(new_df[["date", "downloads"]], CSV_PATH)  # keep only the columns we need
    
    # 3) Compute metrics from the combined dataset
    compute_metrics(combined_df)
    
    # 4) You can add more analytics or printing here
    # For example, print the last few rows
    print("\nLast few rows of the final dataset:")
    print(combined_df.tail())


if __name__ == "__main__":
    main()

