"""LLM clients used by document structuring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import os
import shutil
import subprocess
import time

import requests

from app.config import QwenConfig
from app.ocr_pipeline.utils.logger import get_logger


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class LLMResponse:
    ok: bool
    text: str = ""
    provider: str = ""
    model: str = ""
    error: str = ""
    raw: Optional[Dict[str, Any]] = None


class OllamaGenerateClient:
    """Small adapter around Ollama's local /api/generate endpoint."""

    provider = "ollama"

    def __init__(self, config: QwenConfig) -> None:
        self.config = config

    def is_available(self, autostart: bool = False) -> bool:
        if not self.config.enabled:
            return False
        response = self._get_tags()
        if response is None:
            if not autostart or not self._start_server():
                return False
            response = self._get_tags()
        if response is None or response.status_code != 200:
            return False
        models = [model.get("name", "") for model in response.json().get("models", [])]
        return self._model_available(models)

    def require_available(self) -> None:
        response = self._get_tags()
        if response is None:
            if not self._start_server():
                raise RuntimeError("Ollama is not running and could not be started automatically.")
            response = self._get_tags()
        if response is None:
            raise RuntimeError("Ollama is not running. Start it first.")
        if response.status_code != 200:
            raise RuntimeError(f"Ollama returned HTTP {response.status_code}")
        models = [model.get("name", "") for model in response.json().get("models", [])]
        if not self._model_available(models):
            raise RuntimeError(f"{self.config.model} not found. Run: ollama pull {self.config.model}")

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        if not self.config.enabled:
            return LLMResponse(False, provider=self.provider, model=self.config.model, error="Qwen disabled")

        try:
            self.require_available()
        except RuntimeError as exc:
            return LLMResponse(False, provider=self.provider, model=self.config.model, error=str(exc))

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_ctx": self.config.context_length,
                "num_predict": self.config.max_output_tokens,
                "num_gpu": self.config.num_gpu,
            },
        }
        if system:
            payload["system"] = system

        last_error = ""
        for attempt in range(3):
            try:
                response = requests.post(
                    f"{self.config.base_url}/api/generate",
                    json=payload,
                    timeout=self.config.timeout_sec,
                )
                if response.status_code == 200:
                    raw = response.json()
                    return LLMResponse(
                        True,
                        text=raw.get("response", "").strip(),
                        provider=self.provider,
                        model=self.config.model,
                        raw=raw,
                    )
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
            except Exception as exc:
                last_error = str(exc)
                LOGGER.debug("Ollama call failed (attempt %s): %s", attempt + 1, last_error)
                self._start_server()

            if attempt < 2:
                time.sleep(2)

        return LLMResponse(False, provider=self.provider, model=self.config.model, error=last_error)

    def _model_available(self, models: list[str]) -> bool:
        return self.config.model in models

    def _get_tags(self) -> requests.Response | None:
        try:
            return requests.get(f"{self.config.base_url}/api/tags", timeout=5)
        except requests.RequestException:
            return None

    def _start_server(self) -> bool:
        executable = self._ollama_executable()
        if not executable:
            return False

        env = os.environ.copy()
        env["OLLAMA_HOST"] = "127.0.0.1:11434"
        env["OLLAMA_CONTEXT_LENGTH"] = str(self.config.context_length)
        env["OLLAMA_NUM_PARALLEL"] = "1"
        env["OLLAMA_FLASH_ATTENTION"] = "false"
        if self.config.num_gpu == 0:
            env["OLLAMA_LLM_LIBRARY"] = "cpu"

        log_dir = Path("output") / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "ollama-autostart.out.log"
        stderr_path = log_dir / "ollama-autostart.err.log"

        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
            try:
                subprocess.Popen(
                    [executable, "serve"],
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    env=env,
                    creationflags=creationflags,
                )
            except OSError as exc:
                LOGGER.debug("Failed to start Ollama automatically: %s", exc)
                return False

        for _ in range(30):
            time.sleep(1)
            response = self._get_tags()
            if response is not None and response.status_code == 200:
                return True
        return False

    def _ollama_executable(self) -> str | None:
        configured = os.environ.get("OLLAMA_EXE")
        candidates = [
            configured,
            str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"),
            shutil.which("ollama"),
        ]
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return candidate
        return None

