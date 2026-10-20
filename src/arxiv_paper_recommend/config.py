"""Project paths and lazily read LLM settings without import-time side effects."""

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DATABASE_PATH = DATA_DIR / "documents.db"
INDEX_PATH = DATA_DIR / "documents.faiss"


@dataclass(frozen=True)
class LLMSettings:
	"""Snapshot of configuration read when a generator is constructed."""

	api_key: str | None = None
	base_url: str | None = None
	model: str | None = None

	@classmethod
	def from_env(cls) -> LLMSettings:
		"""Read root .env defaults; process variables take precedence."""
		from dotenv import dotenv_values

		values = {**dotenv_values(PROJECT_ROOT / ".env"), **os.environ}
		return cls(
			api_key=values.get("LLM_API_KEY"),
			base_url=values.get("LLM_API_URL"),
			model=values.get("LLM_MODEL"),
		)
