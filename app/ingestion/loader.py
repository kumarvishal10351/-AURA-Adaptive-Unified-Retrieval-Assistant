import re
from langchain_community.document_loaders import PyMuPDFLoader


def load_pdf(file_path: str):
    """
    Load a PDF and clean page content.

    Strategy:
    - Preserve paragraph breaks (double newlines → two spaces kept as a space).
    - Collapse runs of whitespace/single newlines into a single space so the
      RecursiveCharacterTextSplitter separator '\\n\\n' still has a chance to
      find real section boundaries that PyMuPDF emits as double newlines.
    - Strip leading/trailing whitespace from each page.
    """
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    for doc in documents:
        text = doc.page_content
        # Normalise Windows line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse runs of 3+ newlines to a double newline (paragraph break)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse single newlines (soft wraps) into a space
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        # Collapse multiple spaces/tabs
        text = re.sub(r"[ \t]+", " ", text)
        doc.page_content = text.strip()

    return documents