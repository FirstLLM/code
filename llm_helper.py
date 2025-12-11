"""
LLM Helper - Unified interface for multiple LLM providers.

Default: Ollama (free, local). Set LLM_PROVIDER environment variable to switch.

Supported providers:
- ollama: Free local inference (default)
- google: Google AI Studio (free tier)
- openrouter: OpenRouter (free models available)
- openai: OpenAI API (paid)

Usage:
    from llm_helper import chat
    response = chat("Explain quantum computing simply.")
    print(response)

To switch providers:
    import os
    os.environ["LLM_PROVIDER"] = "google"
    os.environ["GOOGLE_API_KEY"] = "your-key"
"""

import os
import json

# Get provider from environment, default to ollama
PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# Model defaults per provider
DEFAULT_MODELS = {
    "ollama": "llama3.2",
    "google": "gemini-1.5-flash",
    "openrouter": "meta-llama/llama-3.2-3b-instruct:free",
    "openai": "gpt-3.5-turbo"
}


def chat(prompt, system="You are a helpful assistant.", temperature=0.7, model=None):
    """
    Send a message to the LLM and return the response.

    Args:
        prompt: The user's message
        system: System prompt setting the assistant's behavior
        temperature: Creativity (0.0 = deterministic, 1.0 = creative)
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
    import requests

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY environment variable. Get one at https://aistudio.google.com/")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    # Gemini uses a different format - combine system and user prompts
    full_prompt = f"{system}\n\nUser: {prompt}"

    response = requests.post(url, json={
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {"temperature": temperature}
    }, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(f"Google AI error: {response.text}")

    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


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
        temperature: Creativity level
        model: Override default model

    Returns:
        The assistant's response as a string
    """
    model = model or DEFAULT_MODELS.get(PROVIDER)

    if PROVIDER == "ollama":
        return _chat_history_ollama(messages, system, temperature, model)
    elif PROVIDER in ("google", "openrouter", "openai"):
        # These providers support the messages format natively
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
        # Google doesn't support message history the same way, flatten it
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        return _chat_google(history_text, system, temperature, model)

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
