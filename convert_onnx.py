# Filename: create_onnx_model.py
# --- Core Dependencies ---
# Run this on your local machine, not in your deployment environment.
# pip install 'sentence-transformers' 'optimum[onnxruntime]' 'torch'

from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from pathlib import Path

def convert_model_to_onnx(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Downloads a SentenceTransformer model and saves it in the lightweight ONNX format.
    This function uses the Optimum library to handle the conversion.
    """
    # The path where the ONNX model will be saved
    onnx_path = Path(f"onnx/{model_name.split('/')[-1]}")
    
    if onnx_path.exists():
        print(f"ONNX model already exists at: ./{onnx_path}")
        print("Skipping conversion. If you want to re-convert, please delete the directory.")
        return

    print(f"Loading base model '{model_name}' and converting to ONNX...")
    
    # This single command will download the PyTorch model, convert it to ONNX,
    # and load it as an optimized model for inference.
    # The `export=True` flag is crucial as it triggers the conversion.
    ort_model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Saving ONNX model and tokenizer to: ./{onnx_path}")
    # Save the converted ONNX model and the tokenizer to the specified path.
    # This directory will contain model.onnx, tokenizer.json, and other config files.
    ort_model.save_pretrained(onnx_path)
    tokenizer.save_pretrained(onnx_path)

    print("-" * 50)
    print("ONNX conversion complete.")
    print(f"Model and tokenizer files are saved in: ./{onnx_path}")
    print("You can now deploy this directory with the lightweight inference script.")
    print("-" * 50)


if __name__ == "__main__":
    convert_model_to_onnx()

