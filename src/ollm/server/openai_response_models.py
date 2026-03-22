"""OpenAI-compatible Responses API transport models."""

import builtins

from pydantic import BaseModel, ConfigDict, Field


class OpenAIResponseInputContentPartRequestModel(BaseModel):
    """Structured content part for a Responses API input message."""

    model_config = ConfigDict(extra="allow", frozen=True)

    type: str
    text: str | None = None
    image_url: str | None = None
    audio_url: str | None = None
    file_url: str | None = None
    file_id: str | None = None


class OpenAIResponseInputMessageRequestModel(BaseModel):
    """OpenAI-compatible Responses API input message."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = "message"
    role: str
    content: str | list[OpenAIResponseInputContentPartRequestModel]


class OpenAIResponseFunctionCallOutputRequestModel(BaseModel):
    """Tool-result input item for a Responses API request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = "function_call_output"
    call_id: str
    output: str | dict[str, object] | list[object] | int | float | bool | None


class OpenAIResponseFunctionToolRequestModel(BaseModel):
    """Custom function-tool definition for a Responses request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = "function"
    name: str
    description: str | None = None
    parameters: dict[str, object] = Field(default_factory=dict)
    strict: bool = False


class OpenAIResponseFunctionToolChoiceRequestModel(BaseModel):
    """Specific function-tool choice for a Responses request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = "function"
    name: str


class OpenAIResponseCreateRequestModel(BaseModel):
    """OpenAI-compatible create-response request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    input: (
        str
        | list[
            OpenAIResponseInputMessageRequestModel
            | OpenAIResponseFunctionCallOutputRequestModel
        ]
    )
    instructions: str | None = None
    previous_response_id: str | None = None
    stream: bool = False
    max_output_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    seed: int | None = None
    tools: list[OpenAIResponseFunctionToolRequestModel] = Field(default_factory=list)
    tool_choice: str | OpenAIResponseFunctionToolChoiceRequestModel | None = None
    parallel_tool_calls: bool = True


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
    status: str = "completed"
    content: list[OpenAIResponseOutputTextResponseModel]


class OpenAIResponseFunctionCallResponseModel(BaseModel):
    """Function-call output item for a response object."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    type: str = "function_call"
    call_id: str
    name: str
    arguments: str
    status: str = "completed"


class OpenAIResponseIncompleteDetailsResponseModel(BaseModel):
    """Incomplete-response details for the Responses API."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reason: str


class OpenAIResponseTextFormatTypeResponseModel(BaseModel):
    """Concrete text-format descriptor for a response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = "text"


class OpenAIResponseTextFormatResponseModel(BaseModel):
    """Text-format wrapper for the Responses API."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: OpenAIResponseTextFormatTypeResponseModel = Field(
        default_factory=OpenAIResponseTextFormatTypeResponseModel
    )


class OpenAIResponseResponseModel(BaseModel):
    """OpenAI-compatible response object."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    id: str
    object_type: str = Field(
        default="response",
        alias="object",
        serialization_alias="object",
    )
    created_at: int
    status: str
    completed_at: int | None = None
    incomplete_details: OpenAIResponseIncompleteDetailsResponseModel | None = None
    input: (
        str
        | list[
            OpenAIResponseInputMessageRequestModel
            | OpenAIResponseFunctionCallOutputRequestModel
        ]
        | None
    ) = None
    max_output_tokens: int | None = None
    model: str
    output: list[
        OpenAIResponseOutputMessageResponseModel
        | OpenAIResponseFunctionCallResponseModel
    ]
    instructions: str | None = None
    previous_response_id: str | None = None
    tools: list[OpenAIResponseFunctionToolRequestModel] = Field(default_factory=list)
    tool_choice: str | OpenAIResponseFunctionToolChoiceRequestModel | None = None
    parallel_tool_calls: bool = True
    reasoning_effort: str | None = None
    store: bool = False
    temperature: float | None = None
    text: OpenAIResponseTextFormatResponseModel = Field(
        default_factory=OpenAIResponseTextFormatResponseModel
    )
    top_p: float | None = None
    truncation: str = "disabled"
    usage: builtins.object | None = None
    user: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    error: builtins.object | None = None


class OpenAIResponseCreatedEventModel(BaseModel):
    """Streaming event emitted when a response starts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = "response.created"
    response: dict[str, object]
    sequence_number: int


class OpenAIResponseInProgressEventModel(BaseModel):
    """Streaming event emitted when a response remains in progress."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = "response.in_progress"
    response: dict[str, object]
    sequence_number: int


class OpenAIResponseOutputItemAddedEventModel(BaseModel):
    """Streaming event emitted when an output item is opened."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = "response.output_item.added"
    response_id: str
    output_index: int = 0
    item: (
        OpenAIResponseOutputMessageResponseModel
        | OpenAIResponseFunctionCallResponseModel
    )
    sequence_number: int


class OpenAIResponseContentPartAddedEventModel(BaseModel):
    """Streaming event emitted when a content part is opened."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = "response.content_part.added"
    response_id: str
    item_id: str
    output_index: int = 0
    content_index: int = 0
    part: OpenAIResponseOutputTextResponseModel
    sequence_number: int


class OpenAIResponseContentPartDoneEventModel(BaseModel):
    """Streaming event emitted when a content part completes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = "response.content_part.done"
    response_id: str
    item_id: str
    output_index: int = 0
    content_index: int = 0
    part: OpenAIResponseOutputTextResponseModel
    sequence_number: int


class OpenAIResponseOutputTextDeltaEventModel(BaseModel):
    """Streaming event emitted for a text delta."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = "response.output_text.delta"
    response_id: str
    item_id: str
    output_index: int = 0
    content_index: int = 0
    delta: str
    sequence_number: int


class OpenAIResponseOutputTextDoneEventModel(BaseModel):
    """Streaming event emitted when text generation completes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = "response.output_text.done"
    response_id: str
    item_id: str
    output_index: int = 0
    content_index: int = 0
    text: str
    sequence_number: int


class OpenAIResponseOutputItemDoneEventModel(BaseModel):
    """Streaming event emitted when an output item completes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = "response.output_item.done"
    response_id: str
    output_index: int = 0
    item: (
        OpenAIResponseOutputMessageResponseModel
        | OpenAIResponseFunctionCallResponseModel
    )
    sequence_number: int


class OpenAIResponseFunctionCallArgumentsDeltaEventModel(BaseModel):
    """Streaming event emitted for a function-call arguments delta."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = "response.function_call_arguments.delta"
    response_id: str
    item_id: str
    output_index: int = 0
    delta: str
    sequence_number: int


class OpenAIResponseFunctionCallArgumentsDoneEventModel(BaseModel):
    """Streaming event emitted when function-call arguments complete."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = "response.function_call_arguments.done"
    response_id: str
    item_id: str
    output_index: int = 0
    arguments: str
    sequence_number: int


class OpenAIResponseCompletedEventModel(BaseModel):
    """Streaming event emitted when a response completes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = "response.completed"
    response: dict[str, object]
    sequence_number: int


class OpenAIResponseFailedEventModel(BaseModel):
    """Streaming event emitted when a response fails."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = "response.failed"
    response: dict[str, object]
    sequence_number: int


class OpenAIDeletedResponseModel(BaseModel):
    """OpenAI-compatible delete-response payload."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    id: str
    object_type: str = Field(
        default="response",
        alias="object",
        serialization_alias="object",
    )
    deleted: bool = True
