import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


@dataclass(frozen=True)
class LLMSettings:
    provider_name: str
    api_key: str | None
    base_url: str | None
    model: str

    @property
    def enabled(self) -> bool:
        if self.provider_name.lower() == "ollama" and self.base_url and self.model:
            return self.base_url.startswith("http://127.0.0.1") or self.base_url.startswith("http://localhost") or bool(self.api_key)
        return bool(self.api_key and self.base_url and self.model)


def get_llm_settings() -> LLMSettings:
    provider_name = os.getenv("PRD_REVIEW_LLM_PROVIDER", "capmial")
    is_ollama = provider_name.lower() == "ollama"
    return LLMSettings(
        provider_name=provider_name,
        api_key=(
            os.getenv("OLLAMA_API_KEY")
            if is_ollama
            else os.getenv("CAPMIAL_API_KEY")
        )
        or os.getenv("PRD_REVIEW_LLM_API_KEY"),
        base_url=(
            os.getenv("OLLAMA_BASE_URL")
            if is_ollama
            else os.getenv("CAPMIAL_BASE_URL")
        )
        or os.getenv("PRD_REVIEW_LLM_BASE_URL"),
        model=(
            os.getenv("OLLAMA_MODEL")
            if is_ollama
            else os.getenv("CAPMIAL_MODEL")
        )
        or os.getenv("PRD_REVIEW_LLM_MODEL", "gpt-4o-mini"),
    )
