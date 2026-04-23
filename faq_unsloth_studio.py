# -*- coding: utf-8 -*-
"""FAQ_Unsloth_Studio_LocalCSV.ipynb
Updated version with local CSV dataset support.

Model: TechAvenger/MyFaqSolver (Phi-3-mini + LoRA, trained with Unsloth)
Pipeline: Text Input → Tokenizer → Model Inference → Answer Output
Runtime required: GPU (T4 or better) — use Google Colab or Kaggle
"""

# ============================================================
# 📦 Step 1 — Install Dependencies
# ============================================================

import subprocess, sys

result = subprocess.run(
    ["pip", "install", "unsloth[colab-new]", "-q"],
    capture_output=True, text=True
)
print(result.stdout[-500:] if result.stdout else "")
print(result.stderr[-300:] if result.stderr else "")

subprocess.run(["pip", "install", "gradio", "-q"], capture_output=True)
print("✅ Dependencies installed")
# Replace unsloth install with:
subprocess.run(["pip", "install", "transformers", "accelerate", "-q"], capture_output=True)

# ============================================================
# 🔑 Step 2 — HuggingFace Login (optional, model is public)
# ============================================================

import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")  # set as Colab secret or paste here

if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN)
    print("✅ Logged in to HuggingFace")
else:
    print("ℹ️  No token — model is public, continuing without login")


# Step 3 replacement — CPU-safe loader
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID       = "TechAvenger/MyFaqSolver"
MAX_NEW_TOKENS = 256

print(f"Loading model: {MODEL_ID}  (CPU mode — expect slow inference)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype = torch.float32,   # float32 required on CPU
    device_map  = "cpu",
)
model.eval()
print("✅ Model loaded on CPU")


# ============================================================
# 📂 Step 4 — Load LOCAL CSV Dataset
# ============================================================

import pandas as pd

# ── CONFIGURE THIS ──────────────────────────────────────────
# Option A: Upload your CSV to Colab and set the path below
# Option B: Mount Google Drive and point to the file there
CSV_PATH = "data.csv"          # ← CHANGE THIS to your CSV file path

# ── Expected CSV format (one of these column layouts) ───────
# Layout 1: columns  ["question", "answer"]
# Layout 2: columns  ["instruction", "output"]
# Layout 3: columns  ["Question", "Answer"]    (case-insensitive handled below)
# Layout 4: single   ["text"]  with "### Question:" / "### Answer:" blocks

QUESTION_COL = "question"   # leave None to auto-detect
ANSWER_COL   = "answer"   # leave None to auto-detect

def load_local_csv(path: str):
    """Load and validate the local FAQ CSV."""
    global QUESTION_COL, ANSWER_COL

    df = pd.read_csv(path)
    print(f"✅ Loaded CSV  : {path}")
    print(f"   Rows        : {len(df)}")
    print(f"   Columns     : {list(df.columns)}")

    # Auto-detect question/answer columns (case-insensitive)
    col_lower = {c.lower(): c for c in df.columns}

    if QUESTION_COL is None:
        for candidate in ["question", "instruction", "query", "input", "q"]:
            if candidate in col_lower:
                QUESTION_COL = col_lower[candidate]
                break

    if ANSWER_COL is None:
        for candidate in ["answer", "output", "response", "a", "label"]:
            if candidate in col_lower:
                ANSWER_COL = col_lower[candidate]
                break

    if QUESTION_COL is None:
        raise ValueError(
            f"Could not find a question column. Columns found: {list(df.columns)}\n"
            "Set QUESTION_COL manually above."
        )

    print(f"   Question col: '{QUESTION_COL}'")
    print(f"   Answer col  : '{ANSWER_COL}' {'(not found — inference only mode)' if ANSWER_COL is None else ''}")
    print("\n── Sample row ──")
    print(df.iloc[0].to_string())
    return df

try:
    dataset_df = load_local_csv(CSV_PATH)
except FileNotFoundError:
    print(f"⚠️  File not found: {CSV_PATH}")
    print("   Upload your CSV to Colab (Files panel on the left) and update CSV_PATH.")
    print("   Continuing with manual fallback questions for now.\n")
    dataset_df = None


# Step 5 replacement — CPU-safe ask()
def ask(question: str, temperature: float = 0.0, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    prompt  = build_prompt(question)
    inputs  = tokenizer(prompt, return_tensors="pt")   # no .to(device) needed

    gen_kwargs = dict(
        max_new_tokens = max_new_tokens,
        pad_token_id   = tokenizer.eos_token_id,
        eos_token_id   = tokenizer.eos_token_id,
    )
    if temperature > 0:
        gen_kwargs["do_sample"]   = True
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["do_sample"]   = False

    with torch.no_grad():
        output_ids = model.generate(**inputs["input_ids"].unsqueeze(0)
                                    if False else model.generate(**inputs, **gen_kwargs))

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ============================================================
# 🧪 Step 6 — Test Single Question
# ============================================================

question = "What was the highest package achieved?"   # ← change this

print(f"❓ Question: {question}")
print("-" * 60)
answer = ask(question)
print(f"💬 Answer:\n{answer}")


# ============================================================
# 📋 Step 7 — Batch Test on LOCAL CSV Samples
# ============================================================

# ── Manual fallback if CSV not loaded ───────────────────────
MANUAL_QUESTIONS = [
    "Which company offered the highest package and for which position?",
    "Which companies hired web developers, and what were their package ranges?",
    "Which student was hired by Sopra Steria, and what was their role?",
    "Who was hired by Yudiz Solution, and what were their respective roles?",
    "What role was offered to Vedant Bharad at Azilen Technologies?",
]

NUM_SAMPLES = 5   # how many rows to test from your CSV

if dataset_df is not None:
    test_rows = dataset_df.head(NUM_SAMPLES)
    test_questions = test_rows[QUESTION_COL].tolist()
    # Ground-truth answers for comparison (if column exists)
    ground_truths  = test_rows[ANSWER_COL].tolist() if ANSWER_COL else [None] * NUM_SAMPLES
    source = f"local CSV ({CSV_PATH})"
else:
    test_questions = MANUAL_QUESTIONS
    ground_truths  = [None] * len(MANUAL_QUESTIONS)
    source = "manual fallback"

print(f"Running batch inference on {len(test_questions)} questions from {source}...\n")

results = []
for i, (q, gt) in enumerate(zip(test_questions, ground_truths), 1):
    print(f"[{i}/{len(test_questions)}] {str(q)[:80]}...")
    pred = ask(str(q), max_new_tokens=150)
    row  = {"Question": q, "Predicted Answer": pred}
    if gt is not None:
        row["Ground Truth"] = gt
    results.append(row)

results_df = pd.DataFrame(results)
print("\n✅ Done! Results:")
pd.set_option("display.max_colwidth", 120)
print(results_df.to_string())


# ============================================================
# 🎛️ Step 8 — Interactive Gradio Chat UI
# ============================================================

import gradio as gr

css = """
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
body, .gradio-container { font-family: 'Sora', sans-serif !important; background: #080a10 !important; color: #e2e8f0 !important; }
.gradio-container { max-width: 820px !important; margin: 0 auto !important; }
#header { text-align:center; padding:2rem 1rem 1.5rem; border-bottom:1px solid #16192a; margin-bottom:1.5rem; }
#header .badge { display:inline-block; background:#0f1220; border:1px solid #252d4a; border-radius:100px; padding:0.3rem 1rem; font-size:0.72rem; color:#6b82c0; margin-bottom:0.8rem; font-family:'JetBrains Mono',monospace; letter-spacing:0.05em; }
#header h1 { font-size:2rem; font-weight:700; letter-spacing:-0.04em; color:#fff; margin-bottom:0.3rem; }
#header p  { font-size:0.85rem; color:#4b5675; }
#chatbox   { background:#0b0d15 !important; border:1px solid #16192a !important; border-radius:16px !important; }
#question-input textarea { background:#0f1118 !important; border:1px solid #1e2338 !important; border-radius:12px !important; color:#e2e8f0 !important; font-family:'Sora',sans-serif !important; font-size:0.91rem !important; padding:0.8rem 1rem !important; min-height:44px !important; }
#question-input textarea:focus { border-color:#3b5bdb !important; box-shadow:0 0 0 3px rgba(59,91,219,0.12) !important; outline:none !important; }
#question-input label span { display:none !important; }
#btn-ask   button { background:#3b5bdb !important; border:none !important; border-radius:12px !important; color:#fff !important; font-weight:600 !important; cursor:pointer !important; }
#btn-ask   button:hover { background:#2f4ac4 !important; }
#btn-clear button { background:#0f1118 !important; border:1px solid #1e2338 !important; border-radius:12px !important; color:#5a6a9a !important; cursor:pointer !important; }
#btn-clear button:hover { background:#151a28 !important; color:#8fa0d0 !important; }
#footer { text-align:center; color:#252d45; font-size:0.74rem; margin-top:1.5rem; font-family:'JetBrains Mono',monospace; }
"""

def chat(question, history):
    if not question.strip():
        return history, ""
    answer = ask(question)
    history.append((question, answer))
    return history, ""

def clear():
    return [], ""

with gr.Blocks(title="FAQ Agent — Unsloth", css=css) as demo:
    gr.HTML("""
    <div id="header">
        <div class="badge">Phi-3-mini · QLoRA · Unsloth · TechAvenger/MyFaqSolver</div>
        <h1>FAQ Answer Agent</h1>
        <p>Powered by your Unsloth fine-tuned model — fast 4-bit inference</p>
    </div>""")

    chatbot = gr.Chatbot(elem_id="chatbox", show_label=False, height=440)

    with gr.Row():
        q_input   = gr.Textbox(
            elem_id    = "question-input",
            placeholder= "Ask your FAQ question here...",
            lines=1, max_lines=4, show_label=False, scale=8,
            interactive=True, autofocus=True
        )
        ask_btn   = gr.Button("Ask →",  elem_id="btn-ask",   variant="primary",   scale=1, min_width=80)
        clear_btn = gr.Button("Clear",  elem_id="btn-clear", variant="secondary", scale=1, min_width=70)

    gr.HTML('<div id="footer">TechAvenger/MyFaqSolver · Phi-3-mini-4k base · LoRA r=16 · Unsloth 4-bit</div>')

    ask_btn.click(fn=chat,  inputs=[q_input, chatbot], outputs=[chatbot, q_input])
    q_input.submit(fn=chat, inputs=[q_input, chatbot], outputs=[chatbot, q_input])
    clear_btn.click(fn=clear, outputs=[chatbot, q_input])

# share=True gives a public URL valid for 72h (useful in Colab)
demo.launch(css=css, share=True)


# ============================================================
# 💾 Step 9 — Export ALL CSV Questions + Predicted Answers
# ============================================================

# Run inference on your entire local CSV and save results
if dataset_df is not None:
    print(f"Running inference on all {len(dataset_df)} rows in your CSV...")
    print("This may take a while — consider running on a subset first.\n")

    export_rows = []
    total = len(dataset_df)
    for idx, row in dataset_df.iterrows():
        q = str(row[QUESTION_COL])
        print(f"[{idx+1}/{total}] {q[:60]}...")
        pred = ask(q)
        export_row = {"question": q, "predicted_answer": pred}
        if ANSWER_COL:
            export_row["ground_truth"] = row[ANSWER_COL]
        export_rows.append(export_row)

    out_df = pd.DataFrame(export_rows)
    out_df.to_csv("faq_results_predicted.csv", index=False)
    print("\n✅ Saved to faq_results_predicted.csv")
    print(out_df.head())

else:
    # Manual fallback export
    my_questions = [
        "What is Section 144 CrPC?",
        "What is the right to information act?",
        "What is the limitation period for filing a civil suit?",
    ]
    rows = [{"question": q, "predicted_answer": ask(q)} for q in my_questions]
    out_df = pd.DataFrame(rows)
    out_df.to_csv("faq_results_predicted.csv", index=False)
    print("✅ Saved fallback results to faq_results_predicted.csv")
    print(out_df)

"""
─────────────────────────────────────────────────────────────────
📝 QUICK SETUP FOR LOCAL CSV
─────────────────────────────────────────────────────────────────

1. Upload your CSV in Colab:
   • Click the 📁 Files icon (left sidebar) → Upload → select your .csv

2. Update CSV_PATH in Step 4:
   CSV_PATH = "your_dataset.csv"   →   CSV_PATH = "placement_data.csv"

3. Supported column layouts (auto-detected):
   ┌────────────────────┬────────────────────┐
   │  Question column   │  Answer column     │
   ├────────────────────┼────────────────────┤
   │  question          │  answer            │
   │  instruction       │  output            │
   │  query / input     │  response / label  │
   │  Q                 │  A                 │
   └────────────────────┴────────────────────┘
   If your columns don't match, set QUESTION_COL and ANSWER_COL manually.

4. If ANSWER_COL exists → batch output includes Ground Truth for comparison.
   If not             → inference-only mode (only predicted answers shown).

─────────────────────────────────────────────────────────────────
📝 Tips
─────────────────────────────────────────────────────────────────
| Topic         | Detail                                                    |
|---------------|-----------------------------------------------------------|
| Speed         | Unsloth 4-bit inference ~2x faster than vanilla HF        |
| Temperature   | 0.0 (default) for consistent FAQ; 0.3-0.7 for variety     |
| Max tokens    | 256 good for FAQ; 512 for detailed legal explanations      |
| Public URL    | share=True in demo.launch() gives a 72h public Gradio link |
| Model source  | Loads merged model from TechAvenger/MyFaqSolver            |
| Local run     | CPU only → ~2-5 min/answer; always use GPU runtime         |
─────────────────────────────────────────────────────────────────
"""