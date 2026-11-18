# Build Your First LLM — companion files

Use the companion supplied with your edition of *[Build Your First LLM](https://leanpub.com/FirstLLM/)*. In this guide **package root** means the extracted directory containing this README, `llm_helper.py`, `data/`, and `notebooks/`. In the author’s checkout only, that directory is `resources/code/`.

## Choose your notebook

The following links open the English notebook files **in this revision**. For the same chapter in another language, open the matching filename in [Türkçe](notebooks/tr/), [Deutsch](notebooks/de/), or [Русский](notebooks/ru/).

| Chapter | Notebook |
|---|---|
| 5 | [Your First Python Program](notebooks/ch05.ipynb) |
| 6 | [NumPy & PyTorch Survival Guide](notebooks/ch06.ipynb) |
| 7 | [Preparing Your Data](notebooks/ch07.ipynb) |
| 8 | [Building the Tokenizer](notebooks/ch08.ipynb) |
| 9 | [The Embedding Layer](notebooks/ch09.ipynb) |
| 10 | [Attention Is All You Need](notebooks/ch10.ipynb) |
| 11 | [Building the Transformer](notebooks/ch11.ipynb) |
| 12 | [Training Your Model](notebooks/ch12.ipynb) |
| 13 | [Fine-Tuning Your Model](notebooks/ch13.ipynb) |
| 14 | [Prompt Engineering](notebooks/ch14.ipynb) |
| 15 | [Building Applications](notebooks/ch15.ipynb) |
| 16 | [Preparing for Production](notebooks/ch16.ipynb) |
| 17 | [Deployment Options](notebooks/ch17.ipynb) |
| 18 | [What’s Next](notebooks/ch18.ipynb) |

## Start a notebook

- **Colab:** extract the download on your computer. Choose **File → Upload notebook** and select the `.ipynb` for your language. Start with its Setup instructions. Upload any additional files through the **Files** pane; uploading a notebook does not upload its neighbouring files. A runtime reset can discard uploads: keep a copy of handoff files on your computer.
- **Jupyter or VS Code:** create a virtual environment with `python -m venv .venv`, activate it, then run `python -m pip install jupyter ipykernel`. Open the notebook with that environment selected as the kernel. Run its own Setup before the examples; use the kernel’s interpreter for installs. Use an empty, writable working directory for each chapter.

For example, Chapter 5 uses `transformers==5.23.0 torch==2.15.0`; Chapter 8 Setup uses `tiktoken==0.14.0 transformers==5.23.0 torch==2.15.0`; Chapters 12–13 use `torch==2.15.0 transformers==5.23.0 tqdm==4.70.1 numpy==2.3.5 matplotlib==3.10.7`. These are chapter-specific requirements, not a reason to install every package before the first exercise. Initial model and dependency downloads may need internet access. Chapter 5 does **not** need the helper, training data, or a provider key.

## Files that cross chapter boundaries

- **7 → 8:** after the Chapter 7 worked pipeline writes `train.jsonl`, copy it into the Chapter 8 kernel’s working directory (in Colab, download it and upload it to the new runtime). Chapter 8 writes `chapter8_output.jsonl`; this tiny exercise is not the default Chapter 12 training corpus.
- **12 → 13:** keep your own `best_model.pt`. Copy or upload it into the Chapter 13 working directory before choosing checkpoint import. Do not load someone else’s untrusted checkpoint.
- **13:** [data/faq_dataset.jsonl](data/faq_dataset.jsonl) is the packaged English FAQ sample. Translated notebooks may generate their own localized FAQ rows.

## Use the helper in Chapters 14–16

Copy [llm_helper.py](llm_helper.py) **from this package root** beside the notebook in the kernel’s working directory. In Colab upload it to `/content/` before the first import; repeat on a fresh runtime. Check `Path.cwd()` from `pathlib` if Python cannot find the file.

Choose **one** provider before importing the helper. This example selects local Ollama (requires a running local service and model for generation):

```python
import os
os.environ["LLM_PROVIDER"] = "ollama"
from llm_helper import chat
```

For `google`, `openrouter`, or `openai`, choose that provider **instead** of `ollama` and configure its corresponding `GOOGLE_API_KEY`, `OPENROUTER_API_KEY`, or `OPENAI_API_KEY` through your environment or runtime secret store. Never put keys in saved cells or outputs. The helper reads the provider at import time: to change providers, restart the kernel, restore any missing files, then select the provider before importing again. See Appendix E for provider-specific setup and costs.

## Contributing

If you spot an issue or have an improvement, please open an issue or pull request.

## License

The companion code is provided under the [MIT License](LICENSE). The book text and PDFs are not included in that grant.
