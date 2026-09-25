from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_host: str = "localhost"
    database_port: int = 5432
    database_name: str = "faq"
    database_user: str = "faq"
    database_password: SecretStr = SecretStr("faq")
    database_connect_timeout: int = Field(default=10, ge=1)
    openrouter_api_key: SecretStr = SecretStr("")
    llm_provider: Literal["openrouter", "ollama"] = "openrouter"
    llm_base_url: str | None = None
    llm_api_key: SecretStr | None = None
    llm_model: str = "deepseek/deepseek-v4-flash-0731"
    llm_temperature: float = 0
    llm_timeout_seconds: float = Field(default=45, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_require_parameters: bool = True
    structured_output_method: str = "json_schema"
    embedding_provider: Literal["openrouter", "ollama"] = "openrouter"
    embedding_base_url: str | None = None
    embedding_api_key: SecretStr | None = None
    embedding_model: str = "qwen/qwen3-embedding-8b"
    embedding_dimension: int = Field(default=4096, gt=0)
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "qwen3:8b"
    ollama_embedding_model: str = "nomic-embed-text"
    embedding_batch_size: int = Field(default=32, ge=1, le=256)
    train_data_path: Path = Path("temp/data/train_our.jsonl")
    chunk_chars: int = Field(default=1800, ge=200)
    chunk_overlap: int = Field(default=200, ge=0)
    search_top_k: int = Field(default=6, ge=1, le=20)
    max_cosine_distance: float | None = Field(default=None, ge=0, le=2)
    max_tool_calls: int = Field(default=3, ge=1, le=10)
    max_evidence_chunks: int = Field(default=8, ge=1, le=20)
    history_turns: int = Field(default=6, ge=0, le=30)
    history_chars: int = Field(default=12000, ge=0)
    max_input_chars: int = Field(default=4000, ge=1, le=20000)
    request_timeout_seconds: float = Field(default=120, gt=0)
    prompts_path: Path = Path(__file__).parents[1] / "application" / "prompts"
    langfuse_enabled: bool = False
    langfuse_base_url: str = "https://cloud.langfuse.com"
    langfuse_public_key: str = ""
    langfuse_secret_key: SecretStr = SecretStr("")
    log_level: str = "INFO"

    @model_validator(mode="after")
    def validate_options(self):
        if self.chunk_overlap >= self.chunk_chars:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_CHARS")
        if self.structured_output_method not in {"json_schema", "function_calling"}:
            raise ValueError("STRUCTURED_OUTPUT_METHOD: json_schema or function_calling")
        if self.langfuse_enabled and not (
            self.langfuse_public_key and self.langfuse_secret_key.get_secret_value()
        ):
            raise ValueError("Langfuse keys are required when LANGFUSE_ENABLED=true")
        return self

    @property
    def chat_base_url(self) -> str:
        return self.llm_base_url or (
            self.ollama_base_url
            if self.llm_provider == "ollama"
            else "https://openrouter.ai/api/v1"
        )

    @property
    def embeddings_base_url(self) -> str:
        return self.embedding_base_url or (
            self.ollama_base_url
            if self.embedding_provider == "ollama"
            else "https://openrouter.ai/api/v1"
        )

    @property
    def embedding_profile(self) -> dict:
        return {
            "provider": self.embedding_provider,
            "url": self.embeddings_base_url.rstrip("/"),
            "model": self.ollama_embedding_model
            if self.embedding_provider == "ollama"
            else self.embedding_model,
            "dimension": self.embedding_dimension,
            "chunk_chars": self.chunk_chars,
            "overlap": self.chunk_overlap,
            "format_version": 1,
        }
