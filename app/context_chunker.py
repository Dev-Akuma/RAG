import spacy
from typing import List

# Load English model once globally for efficiency
nlp = spacy.load("en_core_web_sm")

def chunk_by_sentences(text: str, max_tokens: int = 150) -> List[str]:
    """
    Split text into coherent chunks based on full sentences.
    Each chunk contains up to max_tokens words (not spaCy tokens).

    Args:
        text (str): The input text.
        max_tokens (int): Maximum number of words per chunk.

    Returns:
        List[str]: A list of chunks.
    """
    if not text.strip():
        return []

    doc = nlp(text)
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sent in doc.sents:
        sent_text = sent.text.strip()

        # Skip empty or extremely short sentences
        if not sent_text or len(sent_text.split()) < 3:
            continue

        sent_tokens = len(sent_text.split())

        # If adding the sentence stays within limit
        if current_tokens + sent_tokens <= max_tokens:
            current_chunk.append(sent_text)
            current_tokens += sent_tokens
        else:
            # Finish the current chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            # Start a new chunk
            current_chunk = [sent_text]
            current_tokens = sent_tokens

    # Final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    print(f"📑 Chunked into {len(chunks)} segments (max_tokens={max_tokens})")
    return chunks
