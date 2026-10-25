"""
Client for interacting with local Ollama inference service.
Provides graceful fallbacks when Ollama is offline or uninstalled.
"""

from typing import Any, Dict, List, Optional
import requests
from threatmap.logging_config import setup_logger

logger = setup_logger("threatmap.llm_client")


class OllamaClient:
    """Interacts with local Ollama REST API for zero-cloud LLM classification."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "llama3:8b",
        temperature: float = 0.0,
        timeout_seconds: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    def is_available(self) -> bool:
        """Checks if local Ollama daemon is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
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
        json_format: bool = True,
    ) -> Optional[str]:
        """
        Sends generation request to Ollama. Returns response text or None if failed.
        """
        if not self.is_available():
            logger.debug("Ollama is not reachable.")
            return None

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
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
                return result.get("response")
            else:
                logger.warning(f"Ollama returned HTTP {resp.status_code}: {resp.text}")
                return None
        except requests.exceptions.Timeout:
            logger.warning(f"Ollama request timed out after {self.timeout_seconds} seconds.")
            return None
        except Exception as e:
            logger.warning(f"Ollama generation failed: {e}")
            return None
