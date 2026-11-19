"""
LLM Helper - Unified interface for multiple LLM providers.

Default: Ollama (free, local). Set LLM_PROVIDER environment variable to switch.

Supported providers:
- ollama: Free local inference (default)
- google: Google AI Studio (free tier)
- openrouter: OpenRouter (free models available)
- openai: OpenAI API (paid)

Usage (choose one provider before importing this module):
    import os
    os.environ["LLM_PROVIDER"] = "ollama"
    from llm_helper import chat
    response = chat("Explain quantum computing simply.")
    print(response)

For another provider, start a fresh process or kernel, configure its key
through your environment, and select that provider before the import.
"""

import os
import json

# Get provider from environment, default to ollama
PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# Model defaults per provider (verified 2026-11; see Appendix E)
# Models and prices change monthly. If a default stops working, check the
# provider's model list (Appendix E shows where) and override via model=...
DEFAULT_MODELS = {
    "ollama": "lfm3:latest",
    "google": "gemini-3.5-flash-lite",
    "openrouter": "openrouter/free",
    "openai": "gpt-6-luna"
}


def chat(prompt, system="You are a helpful assistant.", temperature=0.7, model=None):
    """
    Send a message to the LLM and return the response.

    Args:
        prompt: The user's message
        temperature: Sampling setting, not a portable unit: each provider
            maps it differently (ranges and defaults differ), so the same
            value does not mean the same randomness across providers
        model: Override the default model for this provider

    Returns:
        The assistant's response as a string
    """
    model = model or DEFAULT_MODELS.get(PROVIDER)

    if PROVIDER == "ollama":
        return _chat_ollama(prompt, system, temperature, model)
    elif PROVIDER == "google":
        return _chat_google(prompt, system, temperature, model)
    elif PROVIDER == "openrouter":
        return _chat_openrouter(prompt, system, temperature, model)
    elif PROVIDER == "openai":
        return _chat_openai(prompt, system, temperature, model)
    else:
        raise ValueError(f"Unknown provider: {PROVIDER}. Use: ollama, google, openrouter, openai")


def _chat_ollama(prompt, system, temperature, model):
    """Chat using local Ollama server."""
    import requests

    url = os.getenv("OLLAMA_HOST", "http://localhost:11434") + "/api/chat"

    response = requests.post(url, json={
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {"temperature": temperature}
    }, timeout=120)

    if response.status_code != 200:
        raise RuntimeError(f"Ollama error: {response.text}")

    return response.json()["message"]["content"]


def _chat_google(prompt, system, temperature, model):
    """Chat using Google AI Studio (Gemini)."""
    return _google_generate([{"role": "user", "parts": [{"text": prompt}]}],
                            system, temperature, model)


def _google_generate(contents, system, temperature, model):
    """Send a Gemini generateContent request and extract the reply text."""
    import requests

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY environment variable. Get one at https://aistudio.google.com/")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    # Keep roles intact: the system prompt goes in system_instruction and
    # the conversation in contents (role "user"/"model"), never flattened
    # into one blob of text.
    response = requests.post(url, json={
        "system_instruction": {"parts": [{"text": system}]},
        "contents": contents,
        "generationConfig": {"temperature": temperature}
    }, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(f"Google AI error: {response.text}")

    return _google_reply_text(response.json())


def _google_reply_text(data):
    """Extract text from a Gemini response; blocked/empty replies raise."""
    block = data.get("promptFeedback", {}).get("blockReason")
    if block:
        raise RuntimeError(f"Google AI blocked the prompt ({block}). Rephrase your message.")
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError("Google AI returned no candidates (empty response). Try again or rephrase.")
    candidate = candidates[0]
    reason = candidate.get("finishReason", "")
    if reason in ("SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST"):
        raise RuntimeError(f"Google AI blocked the response ({reason}). Rephrase your message.")
    parts = candidate.get("content", {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts)
    if not text.strip():
        raise RuntimeError(f"Google AI returned empty text (finishReason={reason or 'unknown'}). Try again.")
    return text


def _chat_openrouter(prompt, system, temperature, model):
    """Chat using OpenRouter (access to many models)."""
    import requests

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable. Get one at https://openrouter.ai/")

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature
        },
        timeout=60
    )

    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter error: {response.text}")

    return response.json()["choices"][0]["message"]["content"]


def _chat_openai(prompt, system, temperature, model):
    """Chat using OpenAI API."""
    import requests

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY environment variable. Get one at https://platform.openai.com/")

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature
        },
        timeout=60
    )

    if response.status_code != 200:
        raise RuntimeError(f"OpenAI error: {response.text}")

    return response.json()["choices"][0]["message"]["content"]


def chat_with_history(messages, system="You are a helpful assistant.", temperature=0.7, model=None):
    """
    Chat with conversation history (for multi-turn conversations).

    Args:
        messages: List of {"role": "user"|"assistant", "content": "..."} dicts
        system: System prompt
        temperature: Sampling setting (provider-dependent, see chat())
        model: Override default model

    Returns:
        The assistant's response as a string
    """
    model = model or DEFAULT_MODELS.get(PROVIDER)

    if PROVIDER == "ollama":
        return _chat_history_ollama(messages, system, temperature, model)
    elif PROVIDER in ("google", "openrouter", "openai"):
        # OpenAI-compatible providers take messages; Gemini gets its own
        # contents/system_instruction format inside the dispatcher below
        return _chat_history_openai_format(messages, system, temperature, model)
    else:
        raise ValueError(f"Unknown provider: {PROVIDER}")


def _chat_history_ollama(messages, system, temperature, model):
    """Multi-turn chat with Ollama."""
    import requests

    url = os.getenv("OLLAMA_HOST", "http://localhost:11434") + "/api/chat"

    all_messages = [{"role": "system", "content": system}] + messages

    response = requests.post(url, json={
        "model": model,
        "messages": all_messages,
        "stream": False,
        "options": {"temperature": temperature}
    }, timeout=120)

    if response.status_code != 200:
        raise RuntimeError(f"Ollama error: {response.text}")

    return response.json()["message"]["content"]


def _chat_history_openai_format(messages, system, temperature, model):
    """Multi-turn chat for OpenAI-compatible APIs."""
    import requests

    if PROVIDER == "google":
        # Gemini speaks contents with roles: "user" stays, "assistant"
        # becomes "model". System prompt goes in system_instruction.
        contents = [
            {"role": "model" if m["role"] == "assistant" else m["role"],
             "parts": [{"text": m["content"]}]}
            for m in messages
        ]
        return _google_generate(contents, system, temperature, model)

    elif PROVIDER == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    elif PROVIDER == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    all_messages = [{"role": "system", "content": system}] + messages

    response = requests.post(url, headers=headers, json={
        "model": model,
        "messages": all_messages,
        "temperature": temperature
    }, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(f"{PROVIDER} error: {response.text}")

    return response.json()["choices"][0]["message"]["content"]


# Quick test function
def test_connection():
    """Test if the current provider is working."""
    try:
        response = chat("Say 'Connection successful!' and nothing else.", temperature=0)
        print(f"Provider: {PROVIDER}")
        print(f"Response: {response}")
        return True
    except Exception as e:
        print(f"Provider: {PROVIDER}")
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    test_connection()
