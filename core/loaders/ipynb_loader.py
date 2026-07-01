from typing import List, Dict, Any
import nbformat
import os


# Load markdown and code cells from a Jupyter notebook with source metadata.
def load_ipynb_file(ipynb_path: str) -> List[Dict[str, Any]]:
    notebook = nbformat.read(ipynb_path, as_version=4)
    records = []
    filename = os.path.basename(ipynb_path)

    for cell_index, cell in enumerate(notebook.cells):
        cell_type = cell.get("cell_type", "unknown")
        source = cell.get("source", "").strip()

        if not source:
            continue

        metadata = {
            "source": "course",
            "doc_type": "ipynb",
            "filename": filename,
            "cell_index": cell_index,
            "cell_type": cell_type,
            "path": ipynb_path,
        }

        records.append(
            {
                "text": source,
                "metadata": metadata,
            }
        )

    return records