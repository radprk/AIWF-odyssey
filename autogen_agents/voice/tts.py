from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
from typing import Optional


def _ensure_output_path(output_dir: Path, stem: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{stem}.wav"

def _escape_for_shell(text: str) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline([text])
    escaped = text.replace("'", "'\\''")
    return f"'{escaped}'"


def synthesize(
    text: str,
    output_dir: str | Path = "data/audio",
    filename_stem: str = "response",
    model: Optional[str] = None,
    voice: Optional[str] = None,
    command: Optional[str] = None,
) -> Path:
    """
    Generate speech audio using Resemble's Chatterbox.

    Provide one of:
    - command: a CLI command template, e.g. "chatterbox --text {text} --out {out}"
    - a Python package import with a callable "synthesize" API
    """
    output_path = _ensure_output_path(Path(output_dir), filename_stem)

    if command:
        formatted = command.format(
            text=_escape_for_shell(text),
            out=str(output_path),
            model=model or "",
            voice=voice or "",
        )
        subprocess.run(formatted, check=True, shell=True)
        return output_path

    try:
        chatterbox = importlib.import_module("chatterbox")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Chatterbox is not installed. Install it from https://github.com/resemble-ai/chatterbox "
            "or pass a CLI command via --tts-cmd."
        ) from exc

    if hasattr(chatterbox, "synthesize"):
        chatterbox.synthesize(
            text=text,
            output_path=str(output_path),
            model=model,
            voice=voice,
        )
        return output_path

    raise RuntimeError(
        "Chatterbox import succeeded but no compatible API was found. "
        "Provide a CLI command via --tts-cmd."
    )
