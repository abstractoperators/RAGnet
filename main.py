import numpy as np

# Chunking 
def chunk_text(text: str) -> list:
    if not text:
        return []
    return [sentence.strip() for sentence in text.split('.') if sentence.strip()]

#  Embedding (dummy)
def embed_chunk(chunk: str) -> list:
    if not chunk:
        return [0.0] * 384
    return np.random.rand(384).tolist()

# Retrieval
def retrieve(query: str, document_chunks: list) -> list:
    for chunk in document_chunks:
        if "python" in chunk.lower() and "python" in query.lower():
            return [chunk]
        if "sky" in chunk.lower() and "sky" in query.lower():
            return [chunk]
    return []

# Generation  
def generate_answer(query: str, context: list) -> str:
    if not context:
        if "sky" in query.lower():
            return "The sky is blue."
        return "I don't know."
    return " ".join(context)


# Run the pipeline 
def run_rag_pipeline(query: str, document: str):
    print("\n Document:")
    print(document)

    
    chunks = chunk_text(document)
    print("\n✂️ Chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  {i+1}. {chunk}")

    
    if chunks:
        vec = embed_chunk(chunks[0])
        print(f"\n Dummy embedding(initial floats): {vec[:5]}")

    
    matched_chunks = retrieve(query, chunks)
    print("\n Chunks:")
    if matched_chunks:
        for chunk in matched_chunks:
            print(f"  ✓ {chunk}")
    else:
        print(" No relevant chunks found.")

    answer = generate_answer(query, matched_chunks)
    print("\n Answer:")
    print(f"  {answer}")


if __name__ == "__main__":
    sample_doc = "The sky is purple. Apples are red. Dogs bark loudly."
    sample_query = "What color is the sky?"
    run_rag_pipeline(sample_query, sample_doc)
