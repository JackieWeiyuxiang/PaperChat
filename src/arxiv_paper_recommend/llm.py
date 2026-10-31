"""Shared structured generation and explicit LLM client ownership."""

from typing import Literal

from openai import OpenAI, omit
from pydantic import BaseModel

from .common.lifecycle import ManagedResource
from .config import LLMSettings


class StructuredGenerator(ManagedResource):
	"""Reuse a structured-output client; injected clients remain caller-owned."""

	def __init__(
		self, *, model: str | None = None, client: OpenAI | None = None
	) -> None:
		super().__init__()
		settings = LLMSettings.from_env()
		selected_model = settings.model if model is None else model
		if not isinstance(selected_model, str) or not selected_model.strip():
			raise ValueError("A non-empty generation model must be configured.")
		self.model = selected_model.strip()
		self.client = (
			client
			if client is not None
			else OpenAI(
				api_key=settings.api_key,
				base_url=settings.base_url,
			)
		)
		if client is None:
			self._resources.callback(self.client.close)

	def _generate[Result: BaseModel](
		self,
		request: str,
		instructions: str,
		result_model: type[Result],
		*,
		reasoning_effort: Literal["none", "high"] = "none",
	) -> Result:
		"""Require a parsed response, omitting temperature when thinking is enabled."""
		self._ensure_open()
		response = self.client.responses.parse(
			model=self.model,
			reasoning={"effort": reasoning_effort},
			temperature=0 if reasoning_effort == "none" else omit,
			instructions=instructions,
			input=request,
			text_format=result_model,
		)
		if response.output_parsed is None:
			raise RuntimeError("Generation did not return a structured result.")
		return response.output_parsed
