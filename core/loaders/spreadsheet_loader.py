from typing import List, Dict, Any
import pandas as pd
import os


# Load CSV or Excel data and return one text record per sheet.
def load_spreadsheet_file(file_path: str) -> List[Dict[str, Any]]:
    extension = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)
    records = []

    # Try UTF-8 first and use Latin-1 as a fallback for CSV files.
    if extension == ".csv":
        try:
            sheets = {"csv": pd.read_csv(file_path, encoding="utf-8")}
        except UnicodeDecodeError:
            sheets = {"csv": pd.read_csv(file_path, encoding="latin1")}

    elif extension in [".xlsx", ".xls"]:
        sheets = pd.read_excel(file_path, sheet_name=None)

    else:
        return records

    for sheet_name, dataframe in sheets.items():
        # Limit large spreadsheets to the first 300 rows for indexing.
        dataframe = dataframe.head(300)
        text = dataframe.to_string(index=False)

        if not text.strip():
            continue

        metadata = {
            "source": "course",
            "doc_type": "spreadsheet",
            "filename": filename,
            "sheet_name": sheet_name,
            "path": file_path,
        }

        records.append(
            {
                "text": text,
                "metadata": metadata,
            }
        )

    return records