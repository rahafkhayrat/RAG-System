from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    embed_model: str
    top_k: int
    llm_base_url: str
    llm_api_key: str
    llm_model: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            data_dir=Path(os.getenv("RAG_DATA_DIR", "data")),
            embed_model=os.getenv("RAG_EMBED_MODEL", "BAAI/bge-m3"),
            top_k=int(os.getenv("RAG_TOP_K", "5")),
            llm_base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        )

    def has_llm(self) -> bool:
        return bool(self.llm_api_key)
