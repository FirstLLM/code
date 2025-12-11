# Build Your First LLM - Code Repository

Companion code and notebooks for *[Build Your First LLM](https://leanpub.com/FirstLLM/)*. Each notebook mirrors a chapter's runnable code so readers can click and run without typing.

## Quick Links (Colab)

**Part II: Python Essentials**
- Chapter 5: Your First Python Program — [Open in Colab](https://colab.research.google.com/github/FirstLLM/code/blob/main/notebooks/ch05.ipynb)
- Chapter 6: NumPy & PyTorch Survival Guide — [Open in Colab](https://colab.research.google.com/github/FirstLLM/code/blob/main/notebooks/ch06.ipynb)

**Part III: Build Your First LLM**
- Chapter 7: Preparing Your Data — [Open in Colab](https://colab.research.google.com/github/FirstLLM/code/blob/main/notebooks/ch07.ipynb)
- Chapter 8: Building the Tokenizer — [Open in Colab](https://colab.research.google.com/github/FirstLLM/code/blob/main/notebooks/ch08.ipynb)
- Chapter 9: The Embedding Layer — [Open in Colab](https://colab.research.google.com/github/FirstLLM/code/blob/main/notebooks/ch09.ipynb)
- Chapter 10: Attention Is All You Need — [Open in Colab](https://colab.research.google.com/github/FirstLLM/code/blob/main/notebooks/ch10.ipynb)
- Chapter 11: Building the Transformer — [Open in Colab](https://colab.research.google.com/github/FirstLLM/code/blob/main/notebooks/ch11.ipynb)
- Chapter 12: Training Your Model — [Open in Colab](https://colab.research.google.com/github/FirstLLM/code/blob/main/notebooks/ch12.ipynb)

**Part IV: Make It Useful**
- Chapter 13: Fine-Tuning Your Model — [Open in Colab](https://colab.research.google.com/github/FirstLLM/code/blob/main/notebooks/ch13.ipynb)
- Chapter 14: Prompt Engineering — [Open in Colab](https://colab.research.google.com/github/FirstLLM/code/blob/main/notebooks/ch14.ipynb)
- Chapter 15: Building Applications — [Open in Colab](https://colab.research.google.com/github/FirstLLM/code/blob/main/notebooks/ch15.ipynb)

---

## LLM Helper (`llm_helper.py`)

Chapters 14 and 15 use `llm_helper.py`, a unified interface for running LLMs. It defaults to **Ollama** (free, local inference) but supports multiple providers.

### Supported Providers

| Provider | Cost | Setup |
|----------|------|-------|
| Ollama (default) | Free | Local install, no account needed |
| Google AI Studio | Free tier | API key from [aistudio.google.com](https://aistudio.google.com/) |
| OpenRouter | Free models | API key from [openrouter.ai](https://openrouter.ai/) |
| OpenAI | Paid | API key from [platform.openai.com](https://platform.openai.com/) |

### Usage

```python
from llm_helper import chat

# Simple chat (uses Ollama by default)
response = chat("Explain quantum computing simply.")
print(response)

# With custom settings
response = chat(
    "Classify this review: 'Great movie!'",
    system="You are a sentiment classifier.",
    temperature=0.3
)

# Multi-turn conversation
from llm_helper import chat_with_history

messages = [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language."},
    {"role": "user", "content": "What is it used for?"}
]
response = chat_with_history(messages)
```

### Switching Providers

```python
import os

# Use Google AI Studio instead
os.environ["LLM_PROVIDER"] = "google"
os.environ["GOOGLE_API_KEY"] = "your-api-key"

# Use OpenRouter (free models available)
os.environ["LLM_PROVIDER"] = "openrouter"
os.environ["OPENROUTER_API_KEY"] = "your-api-key"
```

See the book's Appendix E for detailed setup instructions for each provider.

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
