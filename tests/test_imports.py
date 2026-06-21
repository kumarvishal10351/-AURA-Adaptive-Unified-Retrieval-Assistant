def test_imports():
    import streamlit
    import langchain
    import faiss

    assert streamlit is not None
    assert langchain is not None
    assert faiss is not None