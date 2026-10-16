"""The two local models, run with onnxruntime on CPU. No PyTorch, no external API.

- Embedder: bge-small-en-v1.5 turns text into 384-number vectors, so questions match
  sections by meaning ("changed my mind" finds the cooling-off period).
- Reader: DistilBERT fine-tuned on SQuAD reads one section and marks the span of text
  that answers the question, with a confidence score.

Run `python scripts/download_models.py` once to fetch the model files into models/.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

MODELS_DIR = Path(os.environ.get("RAG_MODELS_DIR", Path(__file__).resolve().parent.parent / "models"))

# bge models are trained to embed search queries with this instruction in front.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def models_available() -> bool:
    return all((MODELS_DIR / name / f).exists() for name in ("embed", "qa") for f in ("model.onnx", "tokenizer.json"))


def _session(name: str):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
    return ort.InferenceSession(str(MODELS_DIR / name / "model.onnx"), options, providers=["CPUExecutionProvider"])


def _tokenizer(name: str, max_length: int):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(MODELS_DIR / name / "tokenizer.json"))
    tok.enable_truncation(max_length=max_length)
    return tok


class Embedder:
    def __init__(self):
        self.session = _session("embed")
        self.tokenizer = _tokenizer("embed", 512)
        self.tokenizer.enable_padding()

    def encode(self, texts: list[str]) -> np.ndarray:
        encodings = self.tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        hidden = self.session.run(None, {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)})[0]
        cls = hidden[:, 0, :]  # bge uses the [CLS] vector as the sentence embedding
        return cls / np.linalg.norm(cls, axis=1, keepdims=True)

    def encode_query(self, question: str) -> np.ndarray:
        return self.encode([QUERY_PREFIX + question])[0]


@dataclass(frozen=True)
class Span:
    text: str  # the answer span, e.g. "USD 85"
    start: int  # character offsets into the context
    end: int
    confidence: float  # probability of this start and end, 0 to 1


class Reader:
    MAX_ANSWER_TOKENS = 30

    def __init__(self):
        self.session = _session("qa")
        self.tokenizer = _tokenizer("qa", 512)

    def read(self, question: str, context: str) -> Span | None:
        enc = self.tokenizer.encode(question, context)
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        start_logits, end_logits = (x[0] for x in self.session.run(None, {"input_ids": ids, "attention_mask": mask}))

        in_context = np.array([s == 1 for s in enc.sequence_ids])
        if not in_context.any():
            return None
        start_logits = np.where(in_context, start_logits, -1e9)
        end_logits = np.where(in_context, end_logits, -1e9)
        start_p = _softmax(start_logits)
        end_p = _softmax(end_logits)

        # Best (start, end) pair with start <= end and a bounded length.
        scores = np.triu(np.outer(start_p, end_p))
        scores = np.tril(scores, self.MAX_ANSWER_TOKENS - 1)
        s, e = np.unravel_index(int(np.argmax(scores)), scores.shape)
        char_start, char_end = enc.offsets[s][0], enc.offsets[e][1]
        return Span(context[char_start:char_end], char_start, char_end, float(scores[s, e]))


def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max())
    return z / z.sum()


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()


@lru_cache(maxsize=1)
def get_reader() -> Reader:
    return Reader()
