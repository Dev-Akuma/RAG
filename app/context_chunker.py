import spacy
from typing import List

# Load English model once globally
nlp = spacy.load("en_core_web_sm")

def chunk_by_sentences(text: str, max_tokens: int = 100) -> List[str]:
    """
    Split text into coherent chunks based on full sentences.
    Each chunk contains sentences with up to max_tokens words.
    
    Args:
        text (str): Input text to chunk.
        max_tokens (int): Maximum number of tokens per chunk.
    
    Returns:
        List[str]: List of text chunks.
    """
    doc = nlp(text)
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sent in doc.sents:
        sent_text = sent.text.strip()
        sent_tokens = len(sent_text.split())

        if current_tokens + sent_tokens <= max_tokens:
            current_chunk.append(sent_text)
            current_tokens += sent_tokens
        else:
            # Save existing chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            # Start new chunk with current sentence
            current_chunk = [sent_text]
            current_tokens = sent_tokens

    # Append last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
