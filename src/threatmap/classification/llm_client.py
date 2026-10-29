"""
Client for interacting with local Ollama inference service or persistent disk cache.
Supports chat messages, reproducible random seed, disk-backed caching, and plain-text responses.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

from threatmap.logging_config import setup_logger

logger = setup_logger("threatmap.llm_client")


class DiskResponseCache:
    """Persistent on-disk cache for LLM inference calls."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash_key(self, payload: Dict[str, Any]) -> str:
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get(self, payload: Dict[str, Any]) -> Optional[str]:
        key = self._hash_key(payload)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    entry = json.load(f)
                    return entry.get("response")
            except Exception as e:
                logger.warning(f"Failed to read cache file {cache_file}: {e}")
        return None

    def set(self, payload: Dict[str, Any], response: str) -> None:
        key = self._hash_key(payload)
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({"payload": payload, "response": response}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to write cache file {cache_file}: {e}")


class OllamaClient:
    """Interacts with local Ollama REST API or disk cache for reproducible classification."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "llama3.1:8b-instruct-q4_0",
        temperature: float = 0.0,
        seed: int = 42,
        timeout_seconds: int = 60,
        cache_dir: Optional[Path] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.seed = seed
        self.timeout_seconds = timeout_seconds

        default_cache_dir = Path(__file__).resolve().parent.parent.parent.parent / "artifacts" / "llm_cache"
        self.cache = DiskResponseCache(cache_dir or default_cache_dir)

    def is_available(self) -> bool:
        """Checks if local Ollama daemon is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    def list_available_models(self) -> List[str]:
        """Lists models currently installed in local Ollama instance."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                return [m.get("name") for m in data.get("models", [])]
        except Exception as e:
            logger.debug(f"Could not fetch Ollama models: {e}")
        return []

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_format: bool = False,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        use_cache: bool = True,
    ) -> Optional[str]:
        """
        Sends single-turn generation request to Ollama with disk caching.
        """
        temp = self.temperature if temperature is None else temperature
        s = self.seed if seed is None else seed

        cache_payload = {
            "type": "generate",
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt,
            "json_format": json_format,
            "temperature": temp,
            "seed": s,
        }

        if use_cache:
            cached = self.cache.get(cache_payload)
            if cached is not None:
                return cached

        if not self.is_available():
            logger.debug("Ollama is not reachable and request not in disk cache.")
            return None

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temp,
                "seed": s,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt
        if json_format:
            payload["format"] = "json"

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout_seconds,
            )
            if resp.status_code == 200:
                result = resp.json()
                response_text = result.get("response", "")
                if use_cache and response_text:
                    self.cache.set(cache_payload, response_text)
                return response_text
            else:
                logger.warning(f"Ollama returned HTTP {resp.status_code}: {resp.text}")
                return None
        except Exception as e:
            logger.warning(f"Ollama generation failed: {e}")
            return None

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        use_cache: bool = True,
    ) -> Optional[str]:
        """
        Sends multi-turn chat request (required for few-shot and context experiments) with disk caching.
        """
        temp = self.temperature if temperature is None else temperature
        s = self.seed if seed is None else seed

        cache_payload = {
            "type": "chat",
            "model": self.model_name,
            "messages": messages,
            "temperature": temp,
            "seed": s,
        }

        if use_cache:
            cached = self.cache.get(cache_payload)
            if cached is not None:
                return cached

        if not self.is_available():
            logger.debug("Ollama is not reachable and chat request not in disk cache.")
            return None

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temp,
                "seed": s,
            },
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout_seconds,
            )
            if resp.status_code == 200:
                result = resp.json()
                msg = result.get("message", {})
                content = msg.get("content", "")
                if use_cache and content:
                    self.cache.set(cache_payload, content)
                return content
            else:
                logger.warning(f"Ollama chat returned HTTP {resp.status_code}: {resp.text}")
                return None
        except Exception as e:
            logger.warning(f"Ollama chat failed: {e}")
            return None
