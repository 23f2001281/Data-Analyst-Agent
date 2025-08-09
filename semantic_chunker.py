import numpy as np
import re
from pathlib import Path
from typing import List

import onnxruntime as ort
from transformers import AutoTokenizer


class ONNXEmbedder:
    """
    A lightweight class to handle sentence embeddings using a pre-converted ONNX model.
    """
    def __init__(self, onnx_path: Path):
        """
        Initializes the embedder by loading the ONNX model and tokenizer.
        """
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model directory not found at {onnx_path}. "
                "Please run the `create_onnx_model.py` script first."
            )
            
        model_file = onnx_path / "model.onnx"
        self.session = ort.InferenceSession(str(model_file))
        self.tokenizer = AutoTokenizer.from_pretrained(onnx_path)
        print("Lightweight ONNX embedder initialized successfully.")

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generates embeddings for a list of texts using the ONNX model."""
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="np")
        
        # Prepare all required inputs for ONNX
        ort_inputs = {}
        for input_info in self.session.get_inputs():
            input_name = input_info.name
            if input_name in inputs:
                ort_inputs[input_name] = inputs[input_name]
            else:
                # Handle missing inputs with defaults
                if input_name == "attention_mask":
                    ort_inputs[input_name] = np.ones_like(inputs['input_ids'])
                elif input_name == "token_type_ids":
                    ort_inputs[input_name] = np.zeros_like(inputs['input_ids'])
        
        # Run inference with ONNX runtime
        ort_outputs = self.session.run(None, ort_inputs)
        token_embeddings = ort_outputs[0]

        # Perform mean pooling
        attention_mask = ort_inputs.get('attention_mask', np.ones_like(inputs['input_ids']))
        attention_mask_expanded = np.expand_dims(attention_mask, axis=-1)
        sum_embeddings = np.sum(token_embeddings * attention_mask_expanded, axis=1)
        sum_mask = np.clip(attention_mask_expanded.sum(1), a_min=1e-9, a_max=None)
        
        return sum_embeddings / sum_mask

def cosine_similarity(v1, v2):
    """Calculates cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def semantic_chunker(text: str, embedder: ONNXEmbedder, similarity_threshold=0.55, min_chunk_size=50):
    # 1. Split the document into sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not sentences:
        return []

    # 2. Generate embeddings for each sentence
    embeddings = embedder.generate_embeddings(sentences)

    # 3. Use a sliding window approach to ensure no content is lost
    chunks = []
    current_chunk_sentences = [sentences[0]]
    
    # Track all sentences to ensure 100% coverage
    processed_sentences = set()
    processed_sentences.add(0)  # First sentence is always included

    for i in range(1, len(sentences)):
        similarity = cosine_similarity(embeddings[i], embeddings[i-1])

        # More lenient chunking - only create new chunk if similarity is very low
        # and we have a substantial chunk already
        if similarity < similarity_threshold and len(current_chunk_sentences) >= 5:
            chunk_text = " ".join(current_chunk_sentences)
            if len(chunk_text) >= min_chunk_size:
                chunks.append(chunk_text)
                # Mark all sentences in this chunk as processed
                for j in range(i - len(current_chunk_sentences), i):
                    processed_sentences.add(j)
            current_chunk_sentences = [sentences[i]]
        else:
            current_chunk_sentences.append(sentences[i])

    # Add the last remaining chunk
    final_chunk_text = " ".join(current_chunk_sentences)
    if len(final_chunk_text) >= min_chunk_size:
        chunks.append(final_chunk_text)
        # Mark remaining sentences as processed
        for j in range(len(sentences) - len(current_chunk_sentences), len(sentences)):
            processed_sentences.add(j)

    # 4. Ensure 100% coverage by handling any unprocessed sentences
    unprocessed_indices = [i for i in range(len(sentences)) if i not in processed_sentences]
    
    if unprocessed_indices:
        print(f"⚠️  Found {len(unprocessed_indices)} unprocessed sentences, creating additional chunks...")
        
        # Group unprocessed sentences into chunks
        unprocessed_chunks = []
        current_unprocessed = []
        
        for idx in unprocessed_indices:
            current_unprocessed.append(sentences[idx])
            
            # Create chunk when we have enough sentences or reach the end
            if len(current_unprocessed) >= 3 or idx == unprocessed_indices[-1]:
                chunk_text = " ".join(current_unprocessed)
                if len(chunk_text) >= min_chunk_size:
                    unprocessed_chunks.append(chunk_text)
                current_unprocessed = []
        
        # Add unprocessed chunks to the main chunks
        chunks.extend(unprocessed_chunks)

    return chunks

# --- Main Execution ---
if __name__ == "__main__":
    # Define the path to the ONNX model directory created by the setup script
    model_path = Path("onnx/all-MiniLM-L6-v2")

    # Initialize the lightweight embedder
    onnx_embedder = ONNXEmbedder(model_path)

    scraped_text = """
    The solar system consists of the Sun and the objects that orbit it.
    These objects include eight planets, their moons, and various other smaller bodies.
    Jupiter is the largest planet, a gas giant known for its Great Red Spot.
    Unlike Jupiter, Mars is a rocky planet, often called the 'Red Planet' due to its iron oxide-rich surface.
    Early computer programming languages were complex and required deep technical knowledge.
    COBOL, for instance, was designed for business, finance, and administrative systems.
    Modern languages like Python emphasize readability and allow developers to write clear, logical code for small and large-scale projects.
    """

    print("\n" + "="*40 + "\n")
    print("--- Semantically Split Chunks ---")
    final_chunks = semantic_chunker(scraped_text, onnx_embedder)

    for i, chunk in enumerate(final_chunks):
        print(f"--- Chunk {i+1} ---")
        print(chunk)
        print()
