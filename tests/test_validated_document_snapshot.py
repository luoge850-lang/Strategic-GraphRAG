import json
from dataclasses import asdict
import pytest
from strategic_graphrag.document_layer import (
    Cell, Table, Page, Document, DocumentLayerReader, ValidatedSnapshotReader, sha256_file,
)


def fixture(tmp_path):
    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"immutable-test-input")
    reader = DocumentLayerReader()
    cell = Cell("c", 0, 0, "10", "10", header_path=("Revenue",))
    table = Table("t", 0, (cell,), 1, 1, raw_matrix=(("10",),))
    page = Page(1, "1", 100, 100, "Revenue 10", "Revenue 10", tables=[table], config_hash=reader.config_hash)
    doc = Document("source", pdf.name, sha256_file(pdf), 1, [page],
                   config_hash=reader.config_hash, build_id="build_test")
    snapshot = tmp_path / "snapshot.json"
    doc.write_json(snapshot)
    cached = ValidatedSnapshotReader({pdf.name: {"path": snapshot, "snapshot_sha256": sha256_file(snapshot)}}, reader)
    return pdf, snapshot, doc, cached


def test_typed_round_trip_and_reuse_without_pdf_decoding(tmp_path):
    pdf, snapshot, doc, reader = fixture(tmp_path)
    restored = reader.read(pdf, build_id="build_test")
    assert asdict(restored) == asdict(doc)
    assert isinstance(restored.pages[0].tables[0].cells[0], Cell)


@pytest.mark.parametrize("change", ["pdf", "artifact", "build", "parser"])
def test_identity_mismatch_never_falls_back(tmp_path, change):
    pdf, snapshot, doc, reader = fixture(tmp_path)
    if change == "pdf":
        pdf.write_bytes(b"changed-input")
    elif change == "artifact":
        snapshot.write_text(snapshot.read_text() + " ")
    elif change == "parser":
        reader.config_hash = "other-config"
    with pytest.raises(ValueError):
        reader.read(pdf, build_id="other-build" if change == "build" else "build_test")


def test_reordered_or_duplicate_pages_are_rejected(tmp_path):
    _, _, doc, _ = fixture(tmp_path)
    payload = doc.to_dict()
    payload["pages"][0]["physical_page_number"] = 2
    with pytest.raises(ValueError, match="ordered"):
        Document.from_dict(payload)
