"""Batched local Qwen3 embeddings for research queries and document text."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
DEFAULT_QUERY_INSTRUCTION = (
	"Given a research topic, retrieve scientific papers whose core research "
	"covers the aspects."
)
EMBEDDING_DIMENSION = 1024


class Qwen3Vectorizer:
	"""Load one HF model and reuse it to produce normalized float32 vectors.

	Model and revision match the corpus archive; the current query instruction
	differs from the archive metadata. Optional model dependencies are imported
	only on construction; importing this package needs only NumPy.
	"""

	def __init__(
		self,
		model: str | Path = DEFAULT_MODEL,
		*,
		revision: str | None = DEFAULT_REVISION,
		device: str | None = None,
		batch_size: int = 8,
		max_length: int = 32768,
		query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
	) -> None:
		"""Load an HF model ID or local directory, using CUDA if available.

		Set revision=None to use an unpinned model source. Inputs exceeding
		max_length (including the query prompt) are truncated on the right.
		The pinned default model uses cached files first and downloads missing
		files anonymously, independent of the machine's Hugging Face login.
		"""
		if not isinstance(model, (str, Path)):
			raise TypeError("Model must be an HF model ID or a local directory path.")
		if not str(model).strip():
			raise ValueError("Model source must not be empty.")
		if revision is not None and (
			not isinstance(revision, str) or not revision.strip()
		):
			raise ValueError("Revision must be a non-empty string or None.")
		if type(batch_size) is not int or batch_size <= 0:
			raise ValueError("Batch size must be a positive integer.")
		if type(max_length) is not int or not 1 <= max_length <= 32768:
			raise ValueError("Max length must be an integer between 1 and 32768.")
		if not isinstance(query_instruction, str) or not query_instruction.strip():
			raise ValueError("Query instruction must be a non-empty string.")
		if device is not None and (not isinstance(device, str) or not device.strip()):
			raise ValueError("Device must be a non-empty string or None.")

		try:
			import torch
			from transformers import AutoModel, AutoTokenizer
		except ModuleNotFoundError as exc:
			if exc.name not in {"torch", "transformers"}:
				raise
			raise ImportError(
				"Qwen3Vectorizer requires the optional model dependencies. "
				"Install them with: uv sync --extra vectorization"
			) from exc

		self._torch = torch
		try:
			self.device = torch.device(
				device
				if device is not None
				else ("cuda" if torch.cuda.is_available() else "cpu")
			)
		except (RuntimeError, ValueError) as exc:
			raise ValueError(f"Invalid device: {device!r}.") from exc
		if self.device.type == "cuda":
			if not torch.cuda.is_available():
				raise ValueError("CUDA was requested but is not available.")
			if (
				self.device.index is not None
				and self.device.index >= torch.cuda.device_count()
			):
				raise ValueError(f"CUDA device {self.device.index} is not available.")

		self.batch_size = batch_size
		self.max_length = max_length
		self.query_instruction = query_instruction
		model_source = self._resolve_model_source(model, revision)
		local_files_only = Path(model_source).is_dir()
		self.tokenizer = AutoTokenizer.from_pretrained(
			model_source,
			revision=revision,
			local_files_only=local_files_only,
			padding_side="left",
			truncation_side="right",
		)
		self.model = AutoModel.from_pretrained(
			model_source,
			revision=revision,
			local_files_only=local_files_only,
			dtype=torch.float32,
		)
		if self.model.config.hidden_size != EMBEDDING_DIMENSION:
			raise ValueError(
				f"Model must produce {EMBEDDING_DIMENSION}-dimensional embeddings."
			)
		self.model.to(self.device)
		self.model.eval()

	@staticmethod
	def _resolve_model_source(model: str | Path, revision: str | None) -> str:
		"""Ensure every required default-model file exists before local loading."""
		if str(model) != DEFAULT_MODEL or revision != DEFAULT_REVISION:
			return str(model)

		from huggingface_hub import hf_hub_download
		from huggingface_hub.errors import LocalEntryNotFoundError

		for filename in (
			"config.json",
			"tokenizer_config.json",
			"tokenizer.json",
			"merges.txt",
			"vocab.json",
			"model.safetensors",
		):
			try:
				cached_file = hf_hub_download(
					DEFAULT_MODEL,
					filename,
					revision=revision,
					token=False,
					local_files_only=True,
				)
			except LocalEntryNotFoundError:
				cached_file = hf_hub_download(
					DEFAULT_MODEL,
					filename,
					revision=revision,
					token=False,
				)
		return str(Path(cached_file).parent)

	def encode_queries(self, texts: str | list[str]) -> NDArray[np.float32]:
		"""Encode queries with the configured research retrieval instruction.

		A string returns shape (1024,); a list returns shape (N, 1024), including
		(0, 1024) for an empty list. Blank strings raise ValueError.
		"""
		return self._encode(texts, is_query=True)

	def encode_documents(self, texts: str | list[str]) -> NDArray[np.float32]:
		"""Encode raw document texts without adding a prompt.

		A string returns shape (1024,); a list returns shape (N, 1024), including
		(0, 1024) for an empty list. Blank strings raise ValueError.
		"""
		return self._encode(texts, is_query=False)

	def _encode(self, texts: str | list[str], *, is_query: bool) -> NDArray[np.float32]:
		"""Encode distinct texts in length-grouped batches, restoring input order."""
		single = isinstance(texts, str)
		if not single and not isinstance(texts, list):
			raise TypeError("Texts must be a string or a list of strings.")
		inputs = [texts] if single else list(texts)
		for index, text in enumerate(inputs):
			if not isinstance(text, str):
				raise TypeError(f"Text at index {index} must be a string.")
			if not text.strip():
				raise ValueError(
					f"Text at index {index} must not be empty or whitespace."
				)

		unique_inputs = list(dict.fromkeys(inputs))
		if not unique_inputs:
			return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
		positions = {text: index for index, text in enumerate(unique_inputs)}
		order = sorted(
			range(len(unique_inputs)),
			key=lambda index: len(unique_inputs[index]),
			reverse=True,
		)
		vectors = np.empty((len(unique_inputs), EMBEDDING_DIMENSION), dtype=np.float32)
		with self._torch.inference_mode():
			for start in range(0, len(order), self.batch_size):
				batch_indices = order[start : start + self.batch_size]
				batch_texts = [unique_inputs[index] for index in batch_indices]
				if is_query:
					batch_texts = [
						f"Instruct: {self.query_instruction}\nQuery: {text}"
						for text in batch_texts
					]
				batch = self.tokenizer(
					batch_texts,
					padding=True,
					truncation=True,
					max_length=self.max_length,
					return_tensors="pt",
				).to(self.device)
				outputs = self.model(**batch, use_cache=False)
				# The tokenizer uses left padding, so the final token is valid.
				embeddings = outputs.last_hidden_state[:, -1].float()
				embeddings = self._torch.nn.functional.normalize(embeddings, p=2, dim=1)
				del outputs
				vectors[batch_indices] = embeddings.cpu().numpy()

		if single:
			return vectors[0]
		return vectors[[positions[text] for text in inputs]]
