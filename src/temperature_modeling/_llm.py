"""
Thin LLM provider abstraction.

Supports Groq (free) and Anthropic (production) via a single env var:
    LLM_PROVIDER=groq        # default when GROQ_API_KEY is set
    LLM_PROVIDER=anthropic   # when ANTHROPIC_API_KEY is set

Auto-detection order: if LLM_PROVIDER is not set, uses groq if GROQ_API_KEY
is present, otherwise falls back to anthropic.

Public API
----------
complete(system, messages, max_tokens) -> str
    Send a chat completion request. `messages` is a list of
    {"role": "user"|"assistant", "content": str} dicts.
    Returns the assistant reply as a plain string, or "" on failure.

is_available() -> bool
    True if a provider is configured and its package is installed.
"""

import logging
import os
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------

def _provider() -> str:
    explicit = os.environ.get("LLM_PROVIDER", "").lower()
    if explicit in ("groq", "anthropic"):
        return explicit
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "none"


_GROQ_MODEL      = "llama-3.3-70b-versatile"
_ANTHROPIC_MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Groq backend
# ---------------------------------------------------------------------------

def _groq_complete(system: str, messages: list, max_tokens: int) -> str:
    try:
        from groq import Groq
    except ImportError:
        log.error("groq package not installed — run: pip install groq")
        return ""

    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        log.warning("GROQ_API_KEY not set")
        return ""

    all_messages = [{"role": "system", "content": system}] + list(messages)
    try:
        client   = Groq(api_key=key)
        response = client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=all_messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        log.exception("Groq completion failed")
        return ""


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------

def _anthropic_complete(system: str, messages: list, max_tokens: int) -> str:
    try:
        import anthropic
    except ImportError:
        log.error("anthropic package not installed — run: pip install anthropic")
        return ""

    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        log.warning("ANTHROPIC_API_KEY not set")
        return ""

    try:
        client   = anthropic.Anthropic(api_key=key)
        response = client.messages.create(
            model=_ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=list(messages),
        )
        return response.content[0].text.strip()
    except Exception:
        log.exception("Anthropic completion failed")
        return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def complete(
    system: str,
    messages: list,
    max_tokens: int = 500,
) -> str:
    """
    Send a chat completion and return the assistant reply.

    Parameters
    ----------
    system:     system-prompt string
    messages:   list of {"role": "user"|"assistant", "content": str}
    max_tokens: upper bound on the reply length

    Returns "" if no provider is configured or the call fails.
    """
    p = _provider()
    if p == "groq":
        return _groq_complete(system, messages, max_tokens)
    if p == "anthropic":
        return _anthropic_complete(system, messages, max_tokens)
    log.debug("No LLM provider configured (set GROQ_API_KEY or ANTHROPIC_API_KEY)")
    return ""


def is_available() -> bool:
    """True if a provider and its package are both present."""
    p = _provider()
    if p == "groq":
        try:
            import groq  # noqa: F401
            return bool(os.environ.get("GROQ_API_KEY"))
        except ImportError:
            return False
    if p == "anthropic":
        try:
            import anthropic  # noqa: F401
            return bool(os.environ.get("ANTHROPIC_API_KEY"))
        except ImportError:
            return False
    return False


def provider_label() -> str:
    """Human-readable label for the active provider, e.g. 'Groq (Llama 3.3 70B)'."""
    p = _provider()
    if p == "groq":
        return f"Groq ({_GROQ_MODEL})"
    if p == "anthropic":
        return f"Anthropic ({_ANTHROPIC_MODEL})"
    return "No AI provider"
