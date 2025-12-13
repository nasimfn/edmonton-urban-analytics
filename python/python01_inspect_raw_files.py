from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw"

def preview_csv(path: Path, n=5):
    print("\n" + "="*80)
    print(f"FILE: {path.name}")
    print("="*80)
    df = pd.read_csv(path, nrows=n, encoding="utf-8", engine="python")
    print("Columns:")
    for c in df.columns:
        print(f" - {c}")
    print("\nSample rows:")
    print(df.head(n).to_string(index=False))

def main():
    csvs = sorted(RAW.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No .csv files found in {RAW}")

    print(f"Found {len(csvs)} CSV file(s) in: {RAW}")
    for p in csvs:
        preview_csv(p)

if __name__ == "__main__":
    main()
