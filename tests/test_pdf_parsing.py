import pytest
import app
def test_pdf_without_extractable_text_raises_error(monkeypatch):
    class FakePage:
        def extract_text(self):
            return ""

    class FakePdf:
        pages = [FakePage()]

    monkeypatch.setattr(
        app,
        "PdfReader",
        lambda path: FakePdf()
    )

    with pytest.raises(ValueError, match="scanned|image-based|extractable"):
        app.process_pdf(1, "empty_bill.pdf")

import pickle
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import app


def test_valid_pdf_is_processed_successfully(monkeypatch, tmp_path):
    # 1. Simulate a PDF containing readable text
    class FakePage:
        def extract_text(self):
            return "This is a sample electricity bill."

    class FakePdf:
        pages = [FakePage()]

    monkeypatch.setattr(
        app,
        "PdfReader",
        lambda path: FakePdf()
    )

    # 2. Simulate the text splitter producing two chunks
    class FakeSplitter:
        def create_documents(self, contents):
            return [
                SimpleNamespace(page_content="First chunk"),
                SimpleNamespace(page_content="Second chunk"),
            ]

    monkeypatch.setattr(
        app,
        "RecursiveCharacterTextSplitter",
        lambda **kwargs: FakeSplitter()
    )

    # 3. Simulate embeddings: 2 chunks, 384 dimensions each
    fake_embeddings = np.zeros((2, 384), dtype=np.float32)

    fake_model = Mock()
    fake_model.encode.return_value = fake_embeddings

    monkeypatch.setattr(app, "embed_model", fake_model)

    # 4. Replace FAISS with a lightweight fake index
    class FakeIndex:
        def __init__(self, dimension):
            self.dimension = dimension
            self.add = Mock()

    fake_index = FakeIndex(384)
    fake_faiss = SimpleNamespace(
        IndexFlatL2=Mock(return_value=fake_index),
        write_index=Mock()
    )

    monkeypatch.setattr(app, "faiss", fake_faiss)

    # 5. Use a temporary directory for saved chunks and index
    monkeypatch.setattr(app, "VECTOR_DIR", str(tmp_path))

    # 6. Run the function being tested
    app.process_pdf(123, "sample_bill.pdf")

    # 7. Verify the expected behavior
    fake_model.encode.assert_called_once_with(
        ["First chunk", "Second chunk"]
    )

    fake_faiss.IndexFlatL2.assert_called_once_with(384)

    fake_index.add.assert_called_once()
    added_embeddings = fake_index.add.call_args.args[0]
    assert added_embeddings.shape == (2, 384)

    fake_faiss.write_index.assert_called_once()
    assert fake_faiss.write_index.call_args.args[0] is fake_index

    # Confirm the chunks were actually saved
    chunks_path = tmp_path / "123_chunks.pkl"
    assert chunks_path.exists()

    with open(chunks_path, "rb") as file:
        saved_chunks = pickle.load(file)

    assert saved_chunks == ["First chunk", "Second chunk"]
