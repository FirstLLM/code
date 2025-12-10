# Colab Notebooks

Companion notebooks for *[Build Your First LLM](https://leanpub.com/FirstLLM/)*. Each notebook mirrors a chapter's runnable code so readers can click and run without typing.

## Quick links (Colab)
- Chapter 5: Your First Python Program — [Open in Colab](https://colab.research.google.com/github/FirstLLM/code/blob/main/notebooks/ch05.ipynb)
- Chapter 6: NumPy & PyTorch Survival Guide — [Open in Colab](https://colab.research.google.com/github/FirstLLM/code/blob/main/notebooks/ch06.ipynb)
- Chapter 7: Preparing Your Data — [Open in Colab](https://colab.research.google.com/github/FirstLLM/code/blob/main/notebooks/ch07.ipynb)
- Chapter 8: Building the Tokenizer — [Open in Colab](https://colab.research.google.com/github/FirstLLM/code/blob/main/notebooks/ch08.ipynb)
- Chapter 9: The Embedding Layer — [Open in Colab](https://colab.research.google.com/github/FirstLLM/code/blob/main/notebooks/ch09.ipynb)
- Chapter 10: Attention Is All You Need — [Open in Colab](https://colab.research.google.com/github/FirstLLM/code/blob/main/notebooks/ch10.ipynb)
- Chapter 11: Building the Transformer — [Open in Colab](https://colab.research.google.com/github/FirstLLM/code/blob/main/notebooks/ch11.ipynb)
- Chapter 12: Training Your Model — [Open in Colab](https://colab.research.google.com/github/FirstLLM/code/blob/main/notebooks/ch12.ipynb)
- Chapter 13: Fine-Tuning Your Model — [Open in Colab](https://colab.research.google.com/github/FirstLLM/code/blob/main/notebooks/ch13.ipynb)

## How to run

### Option 1: Google Colab (no install)
1. Click the Colab link above.
2. Run cells top-to-bottom. The first cell installs any needed packages if required.
3. CPU is fine; GPU just speeds heavier demos.

### Option 2: VS Code locally
1. Install VS Code + Python extension.
2. Create and activate a virtual env, then install deps (e.g., transformers for ch05; torch is included in Colab for ch06/ch07). For a fresh env:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install transformers==4.46.1  # for ch05
   ```
3. Open the notebook in VS Code, select the `.venv` kernel, and run cells.

### Option 3: Jupyter locally
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install jupyter transformers==4.46.1
jupyter notebook notebooks/ch05.ipynb  # adjust path/notebook as needed
```

## Contributing

If you spot an issue or have an improvement, please open an issue or pull request.

## License

Code is provided under the MIT License. See `LICENSE` for details.
