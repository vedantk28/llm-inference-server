"""
Application configuration via environment variables.

Uses Pydantic BaseSettings for type-safe, validated config with .env file support.
All settings can be overridden via environment variables.
"""

from enum import Enum
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class BackendType(str, Enum):
    """Supported local inference backends."""
    OLLAMA = "ollama"
    VLLM = "vllm"


class Settings(BaseSettings):
    """
    Central configuration for the LLM Inference Server.
    
    Values are loaded from environment variables and/or a .env file.
    Environment variables take precedence over .env values.
    """

    # --- Backend Selection ---
    active_backend: BackendType = Field(
        default=BackendType.OLLAMA,
        description="Which local inference backend to use",
    )
    default_model: str = Field(
        default="mistral:latest",
        description="Default model to use if not specified in request",
    )

    # --- Ollama Backend ---
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for the Ollama server",
    )

    # --- vLLM Backend ---
    vllm_base_url: str = Field(
        default="http://localhost:8001",
        description="Base URL for the vLLM OpenAI-compatible server",
    )

    # --- Token Budget ---
    max_tokens_budget: int = Field(
        default=4096,
        ge=1,
        description="Maximum total tokens (prompt + completion) allowed per request",
    )

    # --- GPU Routing ---
    gpu_saturation_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="GPU utilization threshold (0.0-1.0) for fallback routing",
    )
    max_concurrent_local: int = Field(
        default=4,
        ge=1,
        description="Max concurrent requests to local backend before Azure fallback",
    )
    gpu_util_cache_ttl: float = Field(
        default=5.0,
        ge=0.0,
        description="Seconds to cache GPU utilization readings from pynvml",
    )

    # --- Azure OpenAI Fallback ---
    azure_openai_endpoint: Optional[str] = Field(
        default=None,
        description="Azure OpenAI endpoint URL",
    )
    azure_openai_api_key: Optional[str] = Field(
        default=None,
        description="Azure OpenAI API key",
    )
    azure_openai_deployment_name: Optional[str] = Field(
        default=None,
        description="Azure OpenAI deployment/model name",
    )
    azure_openai_api_version: str = Field(
        default="2024-12-01-preview",
        description="Azure OpenAI API version",
    )

    # --- Tokenizer ---
    hf_tokenizer_model: Optional[str] = Field(
        default=None,
        description="HuggingFace model name for AutoTokenizer (local models)",
    )

    # --- Server ---
    host: str = Field(default="0.0.0.0", description="Server bind host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server bind port")

    # --- Cost Tracking ---
    # Cost per 1K tokens for each backend (used in benchmarks and metrics)
    local_cost_per_1k_input: float = Field(
        default=0.0,
        ge=0.0,
        description="Cost per 1K input tokens for local inference (usually 0 = free)",
    )
    local_cost_per_1k_output: float = Field(
        default=0.0,
        ge=0.0,
        description="Cost per 1K output tokens for local inference",
    )
    azure_cost_per_1k_input: float = Field(
        default=0.15,
        ge=0.0,
        description="Cost per 1K input tokens for Azure OpenAI",
    )
    azure_cost_per_1k_output: float = Field(
        default=0.60,
        ge=0.0,
        description="Cost per 1K output tokens for Azure OpenAI",
    )

    @property
    def azure_configured(self) -> bool:
        """Check if Azure OpenAI fallback is fully configured."""
        return all([
            self.azure_openai_endpoint,
            self.azure_openai_api_key,
            self.azure_openai_deployment_name,
        ])

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


# Singleton instance — import this throughout the app
settings = Settings()
