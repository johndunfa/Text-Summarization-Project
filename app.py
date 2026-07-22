import os
import re
import socket
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr


def _ensure_nltk() -> Tuple["set[str]", "callable", "callable"]:
    """
    Load NLTK stopwords + tokenizers.

    NLTK 3.8+ uses `punkt_tab` for `sent_tokenize`. We probe tokenization and
    download `punkt`, `punkt_tab`, and `stopwords` if anything is missing.
    """
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize

    def _probe() -> None:
        sent_tokenize("Hello world. Second sentence.")
        _ = stopwords.words("english")

    try:
        _probe()
    except LookupError:
        for pkg in ("punkt", "punkt_tab", "stopwords"):
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
        try:
            _probe()
        except LookupError as e:
            raise RuntimeError(
                "NLTK data is still missing after auto-download.\n\n"
                "Fix (run once in a terminal):\n"
                '  python -c "import nltk; nltk.download(\'punkt\'); nltk.download(\'punkt_tab\'); nltk.download(\'stopwords\')"\n\n'
                f"Underlying error: {e}"
            ) from e

    stop_words = set(stopwords.words("english"))
    return stop_words, sent_tokenize, word_tokenize


STOP_WORDS = set()
sent_tokenize = None
word_tokenize = None


def _get_nltk_handles():
    global STOP_WORDS, sent_tokenize, word_tokenize
    if sent_tokenize is None or word_tokenize is None or not STOP_WORDS:
        STOP_WORDS, sent_tokenize, word_tokenize = _ensure_nltk()
    return STOP_WORDS, sent_tokenize, word_tokenize


def normalize_whitespace(text: str) -> str:
    text = (text or "").replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def clean_token(tok: str) -> str:
    tok = (tok or "").lower()
    tok = re.sub(r"[^a-z0-9']+", "", tok)
    return tok


def tokenize_with_stopwords(text: str):
    text = normalize_whitespace(text)
    stop_words, sent_tok, word_tok = _get_nltk_handles()
    sentences = sent_tok(text) if text else []

    words = []
    for s in sentences:
        for tok in word_tok(s):
            tok = clean_token(tok)
            if tok and tok not in stop_words and len(tok) > 1:
                words.append(tok)
    return sentences, words


def frequency_extractive_summary(text: str, n_sentences: int = 2) -> str:
    sentences, words = tokenize_with_stopwords(text)
    if not sentences:
        return ""
    if not words:
        return " ".join(sentences[:n_sentences])

    stop_words, _, word_tok = _get_nltk_handles()
    word_freq = Counter(words)
    max_freq = max(word_freq.values())
    word_score = {w: f / max_freq for w, f in word_freq.items()}

    sentence_scores = {}
    for i, s in enumerate(sentences):
        tokens = [clean_token(t) for t in word_tok(s)]
        tokens = [t for t in tokens if t and t not in stop_words and len(t) > 1]
        sentence_scores[i] = sum(word_score.get(t, 0.0) for t in tokens)

    top_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:n_sentences]
    top_indices = sorted(top_indices)
    return " ".join(sentences[i] for i in top_indices)


def textrank_extractive_summary(text: str, n_sentences: int = 2) -> str:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.text_rank import TextRankSummarizer
    from sumy.utils import get_stop_words

    text = (text or "").strip()
    if not text:
        return ""

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summarizer.stop_words = get_stop_words("english")
    summary_sentences = summarizer(parser.document, n_sentences)
    return " ".join(str(s) for s in summary_sentences)


@lru_cache(maxsize=2)
def _load_seq2seq(model_id: str):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def transformers_abstractive_summary(
    text: str,
    model_id: str,
    max_new_tokens: int = 80,
    num_beams: int = 4,
    min_new_tokens: int = 20,
) -> str:
    import torch

    text = (text or "").strip()
    if not text:
        return ""

    tokenizer, model, device = _load_seq2seq(model_id)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompt_prefix = "summarize: " if "t5" in model_id.lower() else ""
    prompt_text = (prompt_prefix + text).replace("\n", " ")

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        min_new_tokens=min(min_new_tokens, max_new_tokens - 1) if max_new_tokens > 1 else 1,
        num_beams=num_beams,
        do_sample=False,
        length_penalty=1.0 if "bart" in model_id.lower() else 2.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # `early_stopping` is deprecated / noisy on some Transformers versions
    try:
        with torch.inference_mode():
            outputs = model.generate(**inputs, **gen_kwargs)
    except TypeError:
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def load_text_from_txt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def load_text_from_pdf(path: str) -> str:
    from PyPDF2 import PdfReader

    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def _text_from_upload(file: Optional[str]) -> str:
    if not file:
        return ""
    p = Path(file)
    if not p.exists():
        return ""
    if p.suffix.lower() == ".txt":
        return load_text_from_txt(str(p))
    if p.suffix.lower() == ".pdf":
        return load_text_from_pdf(str(p))
    return ""


def _upload_path_from_gradio(upload) -> Optional[str]:
    """Gradio 6+ File often returns a str path; older versions returned a file-like with .name."""
    if not upload:
        return None
    if isinstance(upload, str):
        return upload.strip() or None
    return getattr(upload, "name", None)


def summarize(
    text: str,
    upload,
    method: str,
    n_sentences: int,
    model_id: str,
) -> str:
    uploaded_text = _text_from_upload(_upload_path_from_gradio(upload))
    final_text = normalize_whitespace(uploaded_text or text)

    if not final_text:
        return "Please paste text or upload a .txt/.pdf file."

    if method == "Frequency (extractive)":
        return frequency_extractive_summary(final_text, n_sentences=n_sentences)

    if method == "TextRank (extractive)":
        return textrank_extractive_summary(final_text, n_sentences=n_sentences)

    max_new_tokens = max(20, int(n_sentences * 25))
    min_new = max(8, int(max_new_tokens * 0.25))
    if min_new >= max_new_tokens:
        min_new = max(1, max_new_tokens - 1)
    return transformers_abstractive_summary(
        final_text,
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new,
    )


def _light_theme_and_css() -> Tuple[gr.themes.Soft, str]:
    theme = gr.themes.Soft(
        primary_hue=gr.themes.Color(
            c50="#eff6ff",
            c100="#dbeafe",
            c200="#bfdbfe",
            c300="#93c5fd",
            c400="#60a5fa",
            c500="#3b82f6",
            c600="#2563eb",
            c700="#1d4ed8",
            c800="#1e40af",
            c900="#1e3a8a",
            c950="#172554",
        ),
        neutral_hue=gr.themes.Color(
            c50="#fafafa",
            c100="#f4f4f5",
            c200="#e4e4e7",
            c300="#d4d4d8",
            c400="#a1a1aa",
            c500="#71717a",
            c600="#52525b",
            c700="#3f3f46",
            c800="#27272a",
            c900="#18181b",
            c950="#09090b",
        ),
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
    ).set(
        body_background_fill="#ffffff",
        block_background_fill="#f8fafc",
        block_border_width="1px",
        block_radius="12px",
        input_background_fill="#ffffff",
        button_primary_background_fill="#2563eb",
        button_primary_background_fill_hover="#1d4ed8",
        button_primary_text_color="#ffffff",
    )
    css = """
    .gradio-container { max-width: 920px !important; margin: auto !important; }
    .hero-card {
        background: linear-gradient(135deg, #ffffff 0%, #eff6ff 55%, #e0f2fe 100%);
        border: 1px solid #bfdbfe;
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.08);
    }
    .hero-card h1 { margin: 0 0 0.35rem 0; font-size: 1.6rem; color: #0f172a; }
    .hero-card p { margin: 0; color: #475569; line-height: 1.55; }
    footer { display: none !important; }
    """
    return theme, css


def summarize_safe(
    text: str,
    upload,
    method: str,
    n_sentences: int,
    model_id: str,
) -> str:
    import traceback

    try:
        return summarize(text, upload, method, n_sentences, model_id)
    except Exception as e:
        return (
            "Could not summarize. Details:\n\n"
            f"{type(e).__name__}: {e}\n\n"
            "---\n"
            f"{traceback.format_exc()}\n\n"
            "Common fixes:\n"
            "• Run once: python -c \"import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')\"\n"
            "• For Transformers: check internet (first model download), disk space, and RAM.\n"
            "• Try **Frequency** or **TextRank** if the neural model fails on your PC."
        )


def build_ui() -> gr.Blocks:
    methods = ["Frequency (extractive)", "TextRank (extractive)", "Transformers (abstractive)"]
    models = ["t5-small", "facebook/bart-large-cnn"]

    with gr.Blocks(title="Text Summarization") as demo:
        gr.HTML(
            """
            <div class="hero-card">
              <h1>Text Summarization Studio</h1>
              <p>
                Bright, simple workspace — paste your paragraph or upload a file, pick a method, and get a short summary.
                <strong>Frequency</strong> and <strong>TextRank</strong> run locally without a large model; <strong>Transformers</strong> uses Hugging Face (first run may download weights).
              </p>
            </div>
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                inp = gr.Textbox(
                    label="Your text",
                    lines=12,
                    placeholder="Paste your article, notes, or document text here…",
                    buttons=["copy"],
                )
                upload = gr.File(
                    label="Optional: upload .txt or .pdf",
                    file_types=[".txt", ".pdf"],
                )
            with gr.Column(scale=2):
                method = gr.Radio(choices=methods, value=methods[0], label="Method")
                n_sentences = gr.Slider(1, 6, value=2, step=1, label="Summary length (approx.)")
                model_id = gr.Dropdown(choices=models, value="t5-small", label="Transformers model")

        out = gr.Textbox(
            label="Summary",
            lines=8,
            buttons=["copy"],
        )
        btn = gr.Button("Summarize", variant="primary", size="lg")
        btn.click(
            fn=summarize_safe,
            inputs=[inp, upload, method, n_sentences, model_id],
            outputs=out,
        )

    return demo


def _find_free_port(host: str = "127.0.0.1", start: int = 7860, attempts: int = 50) -> int:
    """Pick the first port in [start, start+attempts) that is not in use."""
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise OSError(
        f"No free TCP port on {host} in range {start}-{start + attempts - 1}. "
        "Close other Gradio/Python servers or set GRADIO_SERVER_PORT to a free port."
    )


if __name__ == "__main__":
    ui = build_ui()
    theme, css = _light_theme_and_css()
    preferred = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    port = _find_free_port(start=preferred)
    print(f"Starting server on http://127.0.0.1:{port}")
    ui.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        theme=theme,
        css=css,
    )
