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