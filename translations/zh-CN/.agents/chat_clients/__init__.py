"""Reusable chat clients for the course notebooks."""

from .openai_compat import OpenAICompatChatClient, from_env

__all__ = ["OpenAICompatChatClient", "from_env"]
