from pathlib import Path
from tempfile import TemporaryDirectory

from core.chunking import chunk_records
from core.loaders.spreadsheet_loader import load_spreadsheet_file


# Create a temporary CSV file for testing.
def create_sample_csv(csv_path):
    csv_path.write_text(
        "employee,department\n"
        "Alice,Engineering\n"
        "Bob,Finance\n",
        encoding="utf-8",
    )


# Verify spreadsheet extraction, metadata, and chunk creation.
def test_spreadsheet_loader():
    with TemporaryDirectory() as temporary_folder:
        csv_path = Path(temporary_folder) / "sample.csv"
        create_sample_csv(csv_path)

        records = load_spreadsheet_file(str(csv_path))
        chunks = chunk_records(records)

        assert len(records) == 1
        assert chunks

        record = records[0]

        assert "Alice" in record["text"]
        assert "Engineering" in record["text"]
        assert "Bob" in record["text"]
        assert "Finance" in record["text"]

        assert record["metadata"]["source"] == "course"
        assert record["metadata"]["doc_type"] == "spreadsheet"
        assert record["metadata"]["filename"] == "sample.csv"
        assert record["metadata"]["sheet_name"] == "csv"
        assert record["metadata"]["path"] == str(csv_path)

        assert chunks[0]["metadata"]["chunk_strategy"] == "spreadsheet"


if __name__ == "__main__":
    test_spreadsheet_loader()
    print("SPREADSHEET LOADER TEST PASSED")
