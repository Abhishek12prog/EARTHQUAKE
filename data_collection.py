from io import StringIO
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import urlopen

import pandas as pd

from config import QUERY_PARAMS, RAW_DATA_PATH, REQUEST_TIMEOUT_SECONDS, USGS_BASE_URL


def build_data_url():
    query_string = "&".join(f"{key}={value}" for key, value in QUERY_PARAMS.items())
    return f"{USGS_BASE_URL}?{query_string}"


def fetch_earthquake_data(output_path=RAW_DATA_PATH):
    output_file = Path(output_path)
    if output_file.exists():
        print(f"Using cached raw data from {output_file.resolve()}")
        df = pd.read_csv(output_file)
        print(f"Loaded cached dataset with shape {df.shape}")
        return df

    data_url = build_data_url()
    print("Requesting earthquake data from USGS...")
    print(data_url)

    try:
        with urlopen(data_url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            csv_text = response.read().decode("utf-8")
    except HTTPError as exc:
        raise RuntimeError(f"USGS request failed with HTTP status {exc.code}.") from exc
    except URLError as exc:
        raise RuntimeError(
            "Could not reach the USGS API. Check your internet connection and try again."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Data download timed out or failed after {REQUEST_TIMEOUT_SECONDS} seconds."
        ) from exc

    df = pd.read_csv(StringIO(csv_text))
    df.to_csv(output_file, index=False)
    print(f"Saved raw data to {output_file.resolve()} with shape {df.shape}")
    return df


if __name__ == "__main__":
    fetch_earthquake_data()
