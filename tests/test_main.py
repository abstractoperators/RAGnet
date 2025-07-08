from main import chunk_text, embed_chunk, retrieve, generate_answer

# Test chunking and empty input
def test_chunk_text_basic_and_empty():
    assert chunk_text("Hello world. This is a test.") == ["Hello world", "This is a test"]
    assert chunk_text("") == []

# Test embedding gives correct vector
def test_embed_chunk_vector_format():
    vec = embed_chunk("Hello world")
    assert isinstance(vec, list)
    assert len(vec) == 384
    assert all(isinstance(x, float) for x in vec)

    vec_empty = embed_chunk("")
    assert vec_empty == [0.0] * 384

# retrieval works for correct match
def test_retrieve_basic_match():
    doc = ["Python is great"]
    result = retrieve("Tell me about Python", doc)
    assert result == ["Python is great"]

# retrieval fails on no match
def test_retrieve_no_match():
    doc = ["Dogs are barking"]
    result = retrieve("What are spaceships?", doc)
    assert result == []

# Test generation returns string
def test_generate_answer_output_type():
    result = generate_answer("Tell me something", ["This is context"])
    assert isinstance(result, str)
    assert len(result) > 0

# Test correct answer from a one relevant fact
def test_single_fact_retrieval():
    doc = ["The sky is purple"]
    result = retrieve("What color is the sky?", doc)
    assert result == ["The sky is purple"]
    answer = generate_answer("What color is the sky?", result)
    assert "purple" in answer.lower()

# No hallucination if no matching fact exists
def test_no_fact_no_hallucination():
    doc = ["Dogs bark", "Apples are red"]
    result = retrieve("What color is the sky?", doc)
    assert result == []
    answer = generate_answer("What color is the sky?", result)
    assert answer.lower() in ["i don't know.", "the sky is blue."]
