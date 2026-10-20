"""README.md and MODEL_CARD.md must be exactly what their templates and results/ give: no number typed by hand."""

from pathlib import Path

from src.report import DOCUMENTS, collect, render


def test_documents_match_the_results():
    values = collect(Path("results"), Path("app/model/card.json"))
    for document, template in DOCUMENTS.items():
        assert Path(document).read_text() == render(Path(template), values), \
            f"{document} is out of date: run python -m src.report"
