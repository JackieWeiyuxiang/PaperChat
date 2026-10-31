"""Shared structured generation and explicit LLM client ownership."""

import time
from collections.abc import Callable
from json import JSONDecodeError
from typing import Literal

from openai import APIResponseValidationError, OpenAI, OpenAIError, omit
from pydantic import BaseModel, ValidationError

from .common.lifecycle import ManagedResource
from .config import LLMSettings

MAX_LLM_RETRIES = 3


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
				max_retries=0,
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
		result_validator: Callable[[Result], Result] | None = None,
	) -> Result:
		"""Share one retry budget across API, parsing, and result validation failures."""
		self._ensure_open()
		client = self.client.with_options(max_retries=0)
		current_instructions = instructions
		for attempt in range(MAX_LLM_RETRIES + 1):
			try:
				response = client.responses.parse(
					model=self.model,
					reasoning={"effort": reasoning_effort},
					temperature=0 if reasoning_effort == "none" else omit,
					instructions=current_instructions,
					input=request,
					text_format=result_model,
				)
			except (
				ValidationError,
				JSONDecodeError,
				APIResponseValidationError,
			) as exc:
				if attempt == MAX_LLM_RETRIES:
					raise
				current_instructions = self._validation_instructions(instructions, exc)
				continue
			except OpenAIError:
				if attempt == MAX_LLM_RETRIES:
					raise
				time.sleep(2**attempt)
				continue

			try:
				parsed = response.output_parsed
				if parsed is None:
					raise RuntimeError("Generation did not return a structured result.")
				return parsed if result_validator is None else result_validator(parsed)
			except (ValidationError, RuntimeError) as exc:
				if attempt == MAX_LLM_RETRIES:
					raise
				current_instructions = self._validation_instructions(instructions, exc)

	@staticmethod
	def _validation_instructions(instructions: str, error: Exception) -> str:
		"""Replace prior feedback while retaining all original response constraints."""
		issues = (
			str(
				[
					{"field": issue["loc"], "error": issue["msg"]}
					for issue in error.errors(include_input=False, include_url=False)
				]
			)
			if isinstance(error, ValidationError)
			else str(error)
		)
		return (
			instructions
			+ f"\nThe previous response failed validation: {issues}\n"
			+ "Regenerate the response to fix this error while following all original requirements. "
			"Return only the required data fields, not the JSON Schema. "
			"Do not include schema keywords such as additionalProperties."
		)
