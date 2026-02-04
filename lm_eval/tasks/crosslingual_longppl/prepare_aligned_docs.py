#!/usr/bin/env python3
"""
Prepare aligned documents for Cross-Lingual LongPPL evaluation.

This script:
1. Loads parallel documents (e.g., Eurovoc)
2. Uses a teacher model (Qwen2-72B) to identify key paragraphs in English
3. Aligns paragraphs to target languages using cross-lingual embeddings
4. Saves aligned document data for efficient evaluation

Usage:
    python -m lm_eval.tasks.crosslingual_longppl.prepare_aligned_docs \
        --teacher_model /path/to/Qwen2-72B-Instruct \
        --dataset_dir /path/to/eurovoc \
        --output aligned_docs.json \
        --n_docs 20 \
        --top_k 8

The output JSON can then be used by setting:
    export CROSSLINGUAL_LONGPPL_ALIGNED_DOCS=/path/to/aligned_docs.json
"""

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm


# Validated configuration
LANGUAGES = ["en", "de", "fr", "es", "fi", "sv", "pl", "lt", "hu", "el", "mt", "lv"]
EMBEDDING_MODEL = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
ALIGNMENT_THRESHOLD = 0.5
MIN_PARAGRAPH_LENGTH = 50


def load_embedding_model(model_name: str = EMBEDDING_MODEL):
    """Load embedding model for paragraph alignment."""
    try:
        from sentence_transformers import SentenceTransformer

        print(f"Loading embedding model: {model_name}")
        return SentenceTransformer(model_name)
    except ImportError:
        print("sentence-transformers not installed, using transformers fallback")
        return _load_embedding_fallback(model_name)


def _load_embedding_fallback(model_name: str):
    """Fallback embedding using transformers directly."""
    from transformers import AutoModel, AutoTokenizer

    class EmbedModel:
        def __init__(self, model_name):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            self.model.eval()

        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            embeddings = []
            device = next(self.model.parameters()).device
            iterator = tqdm(texts, disable=not show_progress_bar)

            with torch.no_grad():
                for text in iterator:
                    inputs = self.tokenizer(
                        text[:512],
                        return_tensors="pt",
                        truncation=True,
                        max_length=256,
                        padding=True,
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = self.model(**inputs)

                    # Mean pooling
                    mask = inputs["attention_mask"].unsqueeze(-1).float()
                    emb = (outputs.last_hidden_state * mask).sum(1) / (
                        mask.sum(1) + 1e-8
                    )

                    if normalize_embeddings:
                        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)

                    embeddings.append(emb.cpu().numpy()[0])

            return np.array(embeddings)

    return EmbedModel(model_name)


def load_teacher_model(model_path: str):
    """Load teacher model for key paragraph selection."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading teacher model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Teacher model loaded")
    return model, tokenizer


def split_paragraphs(text: str, min_length: int = MIN_PARAGRAPH_LENGTH) -> List[str]:
    """Split text into paragraphs by double newlines."""
    paragraphs = []
    for para in text.split("\n\n"):
        para = para.strip()
        if len(para) >= min_length:
            paragraphs.append(para)
    return paragraphs


def compute_paragraph_ppl(
    model, tokenizer, paragraph: str, max_length: int = 2048
) -> float:
    """Compute perplexity for a single paragraph using the teacher model."""
    import torch.nn.functional as F

    encoded = tokenizer(paragraph, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encoded["input_ids"].to(model.device)

    if input_ids.shape[1] < 10:
        return None

    with torch.no_grad():
        outputs = model(input_ids)

    shift_logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    ppl = torch.exp(loss).float().item()

    return ppl if ppl < 100 else None


def align_paragraphs(
    en_embeddings: np.ndarray,
    tgt_embeddings: np.ndarray,
    threshold: float = ALIGNMENT_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Align English paragraphs to target language using cosine similarity."""
    similarities = en_embeddings @ tgt_embeddings.T

    alignments = []
    for i in range(len(en_embeddings)):
        best_j = int(np.argmax(similarities[i]))
        best_sim = float(similarities[i, best_j])

        alignments.append(
            {
                "en_idx": i,
                "tgt_idx": best_j if best_sim >= threshold else None,
                "similarity": best_sim,
            }
        )

    return alignments


def load_dataset(
    dataset_dir: str, languages: List[str], n_docs: int
) -> List[Dict[str, str]]:
    """Load parallel documents from dataset directory."""
    docs = []

    # Try loading from JSON files
    for lang in languages:
        path = os.path.join(dataset_dir, f"eurovoc_{lang}.json")
        if os.path.exists(path):
            with open(path) as f:
                lang_docs = json.load(f)

            for i, doc in enumerate(lang_docs[:n_docs]):
                if i >= len(docs):
                    docs.append({"celex_id": doc.get("celex_id", f"doc_{i}")})
                docs[i][lang] = doc.get("text", "")

    return docs[:n_docs]


def process_documents(
    docs: List[Dict],
    teacher_model,
    teacher_tokenizer,
    embed_model,
    languages: List[str],
    top_k: int = 8,
    alignment_threshold: float = ALIGNMENT_THRESHOLD,
) -> List[Dict]:
    """Process documents to create aligned key paragraph data."""
    aligned_docs = []

    for doc in tqdm(docs, desc="Processing documents"):
        doc_data = {
            "celex_id": doc.get("celex_id", ""),
            "lang_data": {},
        }

        # Get English paragraphs
        en_text = doc.get("en", "")
        if not en_text:
            continue

        en_paras = split_paragraphs(en_text)
        if len(en_paras) < top_k:
            continue

        # Compute PPL for each English paragraph
        en_ppls = []
        for para in tqdm(en_paras, desc="Computing EN PPL", leave=False):
            ppl = compute_paragraph_ppl(teacher_model, teacher_tokenizer, para)
            en_ppls.append(ppl if ppl is not None else 0)

        # Select top-k highest PPL paragraphs as keys
        valid_indices = [(i, p) for i, p in enumerate(en_ppls) if p > 0]
        valid_indices.sort(key=lambda x: x[1], reverse=True)
        key_en_indices = [i for i, _ in valid_indices[:top_k]]

        # Get English embeddings
        en_embeddings = embed_model.encode(en_paras, normalize_embeddings=True)

        # Store English data
        doc_data["lang_data"]["en"] = {
            "paras": en_paras,
            "key_indices": key_en_indices,
            "ppls": en_ppls,
        }

        # Process each target language
        for lang in languages:
            if lang == "en":
                continue

            tgt_text = doc.get(lang, "")
            if not tgt_text:
                continue

            tgt_paras = split_paragraphs(tgt_text)
            if len(tgt_paras) < 3:
                continue

            # Get target embeddings
            tgt_embeddings = embed_model.encode(tgt_paras, normalize_embeddings=True)

            # Align paragraphs
            alignments = align_paragraphs(
                en_embeddings, tgt_embeddings, alignment_threshold
            )

            # Map key English indices to target indices
            key_tgt_indices = []
            for en_idx in key_en_indices:
                alignment = alignments[en_idx]
                if alignment["tgt_idx"] is not None:
                    key_tgt_indices.append(alignment["tgt_idx"])

            # Store target language data
            doc_data["lang_data"][lang] = {
                "paras": tgt_paras,
                "key_indices": key_tgt_indices,
                "alignments": alignments,
            }

        aligned_docs.append(doc_data)

    return aligned_docs


def main():
    parser = argparse.ArgumentParser(
        description="Prepare aligned documents for Cross-Lingual LongPPL"
    )
    parser.add_argument("--teacher_model", required=True, help="Path to teacher model")
    parser.add_argument(
        "--dataset_dir", required=True, help="Path to dataset directory"
    )
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--n_docs", type=int, default=20, help="Number of documents")
    parser.add_argument(
        "--top_k", type=int, default=8, help="Number of key paragraphs"
    )
    parser.add_argument(
        "--embed_model", default=EMBEDDING_MODEL, help="Embedding model"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=ALIGNMENT_THRESHOLD,
        help="Alignment threshold",
    )
    parser.add_argument(
        "--languages", nargs="+", default=LANGUAGES, help="Languages to process"
    )
    args = parser.parse_args()

    # Load models
    teacher_model, teacher_tokenizer = load_teacher_model(args.teacher_model)
    embed_model = load_embedding_model(args.embed_model)

    # Load dataset
    docs = load_dataset(args.dataset_dir, args.languages, args.n_docs)
    print(f"Loaded {len(docs)} documents")

    # Process documents
    aligned_docs = process_documents(
        docs,
        teacher_model,
        teacher_tokenizer,
        embed_model,
        args.languages,
        args.top_k,
        args.threshold,
    )

    # Compute alignment statistics
    total_alignments = 0
    successful_alignments = 0
    for doc in aligned_docs:
        for lang, data in doc["lang_data"].items():
            if lang != "en" and "alignments" in data:
                for a in data["alignments"]:
                    total_alignments += 1
                    if a["tgt_idx"] is not None:
                        successful_alignments += 1

    # Save output
    output = {
        "teacher": args.teacher_model,
        "embed_model": args.embed_model,
        "align_threshold": args.threshold,
        "n_docs": len(aligned_docs),
        "top_k": args.top_k,
        "languages": args.languages,
        "alignment_stats": {
            "total": total_alignments,
            "aligned": successful_alignments,
        },
        "timestamp": datetime.now().isoformat(),
        "doc_data": aligned_docs,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(aligned_docs)} aligned documents to {args.output}")
    if total_alignments > 0:
        print(
            f"Alignment success rate: {successful_alignments}/{total_alignments} "
            f"({100 * successful_alignments / total_alignments:.1f}%)"
        )


if __name__ == "__main__":
    main()
