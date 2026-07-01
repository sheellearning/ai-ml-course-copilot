from pathlib import Path
from tempfile import TemporaryDirectory

from core.ingestion import ingest_folder


# Verify that supported files are loaded recursively and chunked correctly.
def test_folder_ingestion():
    with TemporaryDirectory() as temporary_folder:
        folder = Path(temporary_folder)
        nested_folder = folder / "nested"
        nested_folder.mkdir()

        sample_csv = nested_folder / "sample.csv"
        sample_csv.write_text(
            "topic,description\n"
            "AI,Artificial intelligence\n"
            "ML,Machine learning\n",
            encoding="utf-8",
        )

        # Unsupported files should be ignored during ingestion.
        (folder / "notes.txt").write_text(
            "This file should not be ingested.",
            encoding="utf-8",
        )

        chunks = ingest_folder(folder)

        assert chunks
        assert all(
            chunk["metadata"]["doc_type"] == "spreadsheet"
            for chunk in chunks
        )
        assert all(
            chunk["metadata"]["filename"] == "sample.csv"
            for chunk in chunks
        )
        assert all(
            chunk["metadata"]["chunk_strategy"] == "spreadsheet"
            for chunk in chunks
        )

        combined_text = " ".join(chunk["text"] for chunk in chunks)

        assert "Artificial intelligence" in combined_text
        assert "Machine learning" in combined_text
        assert "This file should not be ingested" not in combined_text


if __name__ == "__main__":
    test_folder_ingestion()
    print("INGESTION TEST PASSED")
