import os
import numpy as np
import pandas as pd
import faiss
import requests
import gradio as gr
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"   # CPU-friendly, ~90 MB
HF_LLM_MODEL   = "mistralai/Mistral-7B-Instruct-v0.3"       # free on HF Inference API
OLLAMA_MODEL   = "llama3.1"                                  # local model via Ollama
OLLAMA_URL     = "http://localhost:11434/api/generate"
FAQ_FILE       = "data.csv"                                   # columns: question, answer
TOP_K          = 3                                           # retrieved FAQ entries per query

# If HF_TOKEN env var is set → HF Spaces mode (Inference API)
# Otherwise               → local dev mode (Ollama)
USE_OLLAMA = os.environ.get("HF_TOKEN") is None

print(f"[mode] {'Local (Ollama)' if USE_OLLAMA else 'HF Spaces (Inference API)'}")

# ---------------------------------------------------------------------------
# Load embedding model (runs on CPU, works on both local + HF free tier)
# ---------------------------------------------------------------------------
print("[loading] Embedding model ...")
embedder = SentenceTransformer(EMBED_MODEL)

# ---------------------------------------------------------------------------
# Load FAQ dataset
# ---------------------------------------------------------------------------
def load_faq(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Create a CSV with columns: question, answer"
        )
    df = pd.read_csv(path)
    if not {"question", "answer"}.issubset(df.columns):
        raise ValueError("faq.csv must have 'question' and 'answer' columns.")
    df = df.dropna(subset=["question", "answer"])
    print(f"[faq] Loaded {len(df)} entries from {path}")
    return df["question"].tolist(), df["answer"].tolist()

questions, answers = load_faq(FAQ_FILE)

# ---------------------------------------------------------------------------
# Build FAISS index (in-memory, rebuilt on startup — fast for ≤500 entries)
# ---------------------------------------------------------------------------
def build_index(qs: list[str]):
    print("[index] Building FAISS index ...")
    embeddings = embedder.encode(qs, show_progress_bar=False).astype("float32")
    faiss.normalize_L2(embeddings)
    idx = faiss.IndexFlatIP(embeddings.shape[1])  # cosine sim via inner product
    idx.add(embeddings)
    print(f"[index] {idx.ntotal} vectors indexed.")
    return idx

index = build_index(questions)

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    q_emb = embedder.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, k)
    return [
        {"question": questions[i], "answer": answers[i], "score": float(s)}
        for s, i in zip(scores[0], indices[0])
        if i >= 0
    ]

# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------
def call_ollama(user_query: str, context_str: str) -> str:
    prompt = (
        "You are a helpful FAQ assistant. "
        "Answer the user's question using only the FAQ context below. "
        "If the context doesn't contain the answer, say so honestly.\n\n"
        f"### FAQ Context:\n{context_str}\n\n"
        f"### User Question:\n{user_query}\n\n"
        "### Answer:"
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        return (
            "⚠ Could not connect to Ollama. "
            "Make sure Ollama is running (`ollama serve`) and the model is pulled "
            f"(`ollama pull {OLLAMA_MODEL}`)."
        )
    except Exception as e:
        return f"⚠ Ollama error: {e}"


def call_hf_api(user_query: str, context_str: str) -> str:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        return "⚠ HF_TOKEN secret is not set. Add it in Space Settings → Secrets."
    try:
        client = InferenceClient(model=HF_LLM_MODEL, token=token)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful FAQ assistant. "
                    "Answer using only the FAQ context provided. "
                    "If the answer isn't there, say so.\n\n"
                    f"### FAQ Context:\n{context_str}"
                ),
            },
            {"role": "user", "content": user_query},
        ]
        resp = client.chat_completion(messages=messages, max_tokens=512, temperature=0.3)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠ HF Inference API error: {e}"

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def answer_query(user_query: str):
    user_query = user_query.strip()
    if not user_query:
        return "", "Please enter a question."

    hits = retrieve(user_query)
    if not hits:
        return "No relevant FAQ entries found.", ""

    context_str = "\n\n".join(
        [f"Q: {h['question']}\nA: {h['answer']}" for h in hits]
    )

    answer = call_ollama(user_query, context_str) if USE_OLLAMA else call_hf_api(user_query, context_str)

    sources_md = "\n".join(
        [f"**{i+1}.** {h['question']}  *(score: {h['score']:.2f})*" for i, h in enumerate(hits)]
    )
    return answer, sources_md

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
MODE_LABEL = "🖥 Local (Ollama)" if USE_OLLAMA else "☁ HF Spaces (Inference API)"

with gr.Blocks(title="FAQ Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# 💬 FAQ Assistant\n*{MODE_LABEL} · {len(questions)} entries indexed*")

    with gr.Row():
        with gr.Column(scale=4):
            query_box = gr.Textbox(
                placeholder="Type your question and press Enter ...",
                label="Your Question",
                lines=2,
                autofocus=True,
            )
        with gr.Column(scale=1, min_width=120):
            submit_btn = gr.Button("Ask ↗", variant="primary", size="lg")

    with gr.Row():
        answer_box = gr.Textbox(
            label="Answer",
            lines=6,
            interactive=False,
            show_copy_button=True,
        )

    with gr.Accordion("📎 Matched FAQ entries", open=False):
        sources_box = gr.Markdown()

    gr.Examples(
        examples=[
            ["What is your return policy?"],
            ["How do I reset my password?"],
            ["Do you offer free shipping?"],
        ],
        inputs=query_box,
    )

    submit_btn.click(answer_query, inputs=query_box, outputs=[answer_box, sources_box])
    query_box.submit(answer_query, inputs=query_box, outputs=[answer_box, sources_box])

if __name__ == "__main__":
    demo.launch(show_error=True)