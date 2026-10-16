"""Download the two local models into models/ (about 100 MB, once).

    python scripts/download_models.py

- bge-small-en-v1.5 (BAAI, MIT): sentence embeddings for semantic search
- distilbert-base-cased-distilled-squad (Hugging Face, Apache 2.0): extractive question answering

Both are the int8-quantised ONNX exports published under the Xenova namespace on
Hugging Face, so they run on CPU with onnxruntime and no PyTorch.
"""

from pathlib import Path

from huggingface_hub import hf_hub_download

MODELS = {
    "embed": ("Xenova/bge-small-en-v1.5", "onnx/model_quantized.onnx"),
    "qa": ("Xenova/distilbert-base-cased-distilled-squad", "onnx/model_quantized.onnx"),
}

ROOT = Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    for name, (repo, onnx_file) in MODELS.items():
        target = ROOT / name
        target.mkdir(parents=True, exist_ok=True)
        for remote, local in [(onnx_file, "model.onnx"), ("tokenizer.json", "tokenizer.json")]:
            if (target / local).exists():
                continue
            path = hf_hub_download(repo, remote)
            (target / local).write_bytes(Path(path).read_bytes())
            print(f"{name}: {local} ({(target / local).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
