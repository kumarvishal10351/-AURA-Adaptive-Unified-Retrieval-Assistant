from langchain_community.document_loaders import PyMuPDFLoader

def load_pdf(file_path: str):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    # Optional cleaning
    for doc in documents:
        doc.page_content = doc.page_content.replace("\n", " ").strip()

    return documents