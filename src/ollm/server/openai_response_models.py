"""OpenAI-compatible Responses API transport models."""

import builtins

from pydantic import BaseModel, ConfigDict, Field


class OpenAIResponseInputTextPartRequestModel(BaseModel):
    """Structured content part for a Responses API input message."""

    model_config = ConfigDict(extra="allow", frozen=True)

    type: str
    text: str | None = None


class OpenAIResponseInputMessageRequestModel(BaseModel):
    """OpenAI-compatible Responses API input message."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str
    content: str | list[OpenAIResponseInputTextPartRequestModel]


class OpenAIResponseCreateRequestModel(BaseModel):
    """OpenAI-compatible create-response request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    input: str | list[OpenAIResponseInputMessageRequestModel]
    instructions: str | None = None
    previous_response_id: str | None = None
    stream: bool = False
    max_output_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    seed: int | None = None


class OpenAIResponseOutputTextResponseModel(BaseModel):
    """Text output content item for a response output message."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = "output_text"
    text: str
    annotations: list[object] = Field(default_factory=list)


class OpenAIResponseOutputMessageResponseModel(BaseModel):
    """Assistant output message for a response object."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    type: str = "message"
    role: str = "assistant"
    content: list[OpenAIResponseOutputTextResponseModel]


class OpenAIResponseResponseModel(BaseModel):
    """OpenAI-compatible response object."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    object: str = "response"
    created_at: int
    status: str
    model: str
    output: list[OpenAIResponseOutputMessageResponseModel]
    instructions: str | None = None
    previous_response_id: str | None = None
    error: builtins.object | None = None


class OpenAIResponseCreatedEventModel(BaseModel):
    """Streaming event emitted when a response starts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = "response.created"
    response: OpenAIResponseResponseModel


class OpenAIResponseOutputTextDeltaEventModel(BaseModel):
    """Streaming event emitted for a text delta."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = "response.output_text.delta"
    response_id: str
    item_id: str
    output_index: int = 0
    content_index: int = 0
    delta: str


class OpenAIResponseOutputTextDoneEventModel(BaseModel):
    """Streaming event emitted when text generation completes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = "response.output_text.done"
    response_id: str
    item_id: str
    output_index: int = 0
    content_index: int = 0
    text: str


class OpenAIResponseCompletedEventModel(BaseModel):
    """Streaming event emitted when a response completes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = "response.completed"
    response: OpenAIResponseResponseModel
