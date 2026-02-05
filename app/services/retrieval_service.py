import os

DATA_PATH = "data/documents/"
VECTOR_PATH = "vector_store/faiss_index"

_model = None

def get_model():
    global _model
    if _model is None:
        print("Loading SentenceTransformer model...")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

documents = []
doc_names = []

def load_documents():
    global documents, doc_names
    if documents: # already loaded
        return
        
    for file in os.listdir(DATA_PATH):
        with open(os.path.join(DATA_PATH, file), "r", encoding="utf-8") as f:
            text = f.read()
            documents.append(text)
            doc_names.append(file)

def create_vector_store():
    import faiss
    import numpy as np
    
    model = get_model()
    embeddings = model.encode(documents)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    faiss.write_index(index, VECTOR_PATH)

    return index

_index = None

def load_vector_store():
    # Always load documents so we have the text mapping
    load_documents()
    
    import faiss
    if os.path.exists(VECTOR_PATH):
        return faiss.read_index(VECTOR_PATH)
    else:
        return create_vector_store()

def get_index():
    global _index
    if _index is None:
        _index = load_vector_store()
    return _index

def search_context(query):
    model = get_model()
    index = get_index()
    
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, 2)

    results = []
    for i in indices[0]:
        if 0 <= i < len(documents):
            results.append(documents[i])
    print("results:", results)
    return "\n".join(results)
