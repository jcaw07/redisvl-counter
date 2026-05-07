# RedisVL downloads tracker

This repo tracks daily PyPI downloads for the `redisvl` package.

The main script fetches all-time daily download data from PyPIStats, filters out mirror downloads, merges the result with the local CSV, deduplicates by date, and writes the updated dataset to `redisvl_downloads.csv`.

## Files

- `redisvl.py`: Fetches fresh RedisVL download data and updates the CSV.
- `redisvl_downloads.csv`: Daily RedisVL downloads, without mirrors.
- `redisvl-charts.py`: Builds a Q3/Q4 summary, charts, and a simple linear forecast from a CSV.
- `requirements.txt`: Python deps for the fetch script.

## Current dataset

As of the latest update, `redisvl_downloads.csv` contains:

- First recorded day: `2024-07-27`
- Last recorded day: `2026-05-06`
- Rows: `647`
- Total downloads: `10,969,092`

Recent monthly totals:

- February 2026: `1,486,619`
- March 2026: `1,858,356`
- April 2026: `1,857,978`
- February-April 2026 total: `5,202,953`

## Run it

Create and activate a virtual environment, then install deps:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Update the download log:

```bash
python redisvl.py
```

The script prints:

- Total downloads in the last 365 days
- Month-over-month growth for the latest 30-day window
- The latest rows in `redisvl_downloads.csv`

## Build charts

Run the chart script with the CSV path:

```bash
python redisvl-charts.py redisvl_downloads.csv
```

The chart script imports `matplotlib` and `scikit-learn`. Install them in the same environment if they are not already available.
