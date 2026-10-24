"""Bilingual user messages: Arabic first, then English."""

from __future__ import annotations

import os
import sys

LANG = os.environ.get("EYE_TRAINER_LANG", "both")  # ar | en | both


def say(ar: str, en: str) -> str:
    if LANG == "ar":
        return ar
    if LANG == "en":
        return en
    return f"{ar}\n  ({en})"


def configure_console() -> None:
    """Make Arabic printable on Windows consoles (cp1252 by default)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
