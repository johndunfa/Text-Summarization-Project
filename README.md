# Text Summarization (Browser App)

This project provides **extractive** and **abstractive** text summarization in a simple **browser UI** (Gradio or Streamlit).

## Streamlit UI (recommended — clean product-style layout)

From `e:\text-summarization-project`:

```powershell
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`). The notebook in this repo is for learning; **Streamlit runs as its own server**, not inside a notebook cell—use a terminal (or `!streamlit run streamlit_app.py` from Jupyter’s terminal).

## Gradio UI (alternative)

```powershell
python -m pip install -r requirements.txt
python app.py
```

Then open the URL printed in the terminal (usually `http://127.0.0.1:7860`). If `7860` is already in use (another Gradio app), the app **picks the next free port** automatically.

To force a starting port:

```powershell
$env:GRADIO_SERVER_PORT="9000"
python app.py
```

## Methods
- **Frequency (extractive)**: picks top-scoring sentences by word frequency
- **TextRank (extractive)**: graph-based sentence ranking (via `sumy`)
- **Transformers (abstractive)**: generates a rewritten summary using a pretrained model (`t5-small` or `facebook/bart-large-cnn`)

## Notes
- The first time you run the Transformers method, it will download model weights (can take a few minutes).
- If NLTK downloads fail (offline), connect to internet once and re-run.

