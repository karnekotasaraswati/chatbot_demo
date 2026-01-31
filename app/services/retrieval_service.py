import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/documents/"
VECTOR_PATH = "vector_store/faiss_index"

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
doc_names = []

def load_documents():
    global documents, doc_names
    for file in os.listdir(DATA_PATH):
        with open(os.path.join(DATA_PATH, file), "r", encoding="utf-8") as f:
            text = f.read()
            documents.append(text)
            doc_names.append(file)

def create_vector_store():
    embeddings = model.encode(documents)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    faiss.write_index(index, VECTOR_PATH)

    return index

index = None

def load_vector_store():
    # Always load documents so we have the text mapping
    load_documents()
    
    if os.path.exists(VECTOR_PATH):
        return faiss.read_index(VECTOR_PATH)
    else:
        return create_vector_store()

index = load_vector_store()

def search_context(query):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, 2)

    results = []
    for i in indices[0]:
        if 0 <= i < len(documents):
            results.append(documents[i])
    print("results:", results)
    return "\n".join(results)
