"""
Hybrid tokenizer supporting both local models (via HuggingFace AutoTokenizer)
and Azure OpenAI models (via tiktoken).

Why hybrid?
  - tiktoken's cl100k_base encoding is accurate for OpenAI models but can be
    10-20% off for models like Qwen, Mistral, or Llama that use their own
    SentencePiece/BPE tokenizers.
  - AutoTokenizer loads the model's exact tokenizer from HuggingFace, giving
    accurate counts (±1 token).
  - We cache tokenizer instances at startup to avoid loading them per-request
    (AutoTokenizer loading can take 1-2 seconds on first call).
"""

import logging
from typing import Optional

import tiktoken

logger = logging.getLogger(__name__)

# Lazy import — transformers is heavy, only load if needed
_AutoTokenizer = None


def _get_auto_tokenizer():
    """Lazy-load transformers.AutoTokenizer to avoid slow startup when not needed."""
    global _AutoTokenizer
    if _AutoTokenizer is None:
        try:
            from transformers import AutoTokenizer

            _AutoTokenizer = AutoTokenizer
        except ImportError:
            logger.warning(
                "transformers not installed — falling back to tiktoken for all models. "
                "Token counts for local models may be ~10-20%% off. "
                "Install with: pip install transformers"
            )
            _AutoTokenizer = False  # Sentinel: tried and failed
    return _AutoTokenizer


class HybridTokenizer:
    """
    Token counter that uses the appropriate tokenizer for each backend.
    
    - For local models (Ollama/vLLM): tries AutoTokenizer first, falls back to tiktoken
    - For Azure OpenAI: uses tiktoken with the model's native encoding
    
    Instances should be created once and reused — tokenizer loading is expensive.
    """

    def __init__(self, model_name: Optional[str] = None, backend: str = "ollama"):
        """
        Args:
            model_name: HuggingFace model ID (e.g. 'mistralai/Mistral-7B-Instruct-v0.3')
                        or OpenAI model name (e.g. 'gpt-4o-mini')
            backend: One of 'ollama', 'vllm', 'azure'
        """
        self._tokenizer = None
        self._mode = "tiktoken_fallback"
        self._model_name = model_name or "unknown"

        if backend in ("ollama", "vllm") and model_name:
            self._init_local_tokenizer(model_name)
        elif backend == "azure" and model_name:
            self._init_azure_tokenizer(model_name)
        else:
            self._init_fallback()

    def _init_local_tokenizer(self, model_name: str) -> None:
        """Try to load the model's native tokenizer from HuggingFace."""
        AutoTokenizer = _get_auto_tokenizer()

        if AutoTokenizer and AutoTokenizer is not False:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,  # Some models (Qwen) need this
                )
                self._mode = "hf"
                logger.info(
                    f"Loaded HuggingFace tokenizer for '{model_name}' — "
                    f"token counts will be accurate"
                )
                return
            except Exception as e:
                logger.warning(
                    f"Failed to load AutoTokenizer for '{model_name}': {e}. "
                    f"Falling back to tiktoken (counts may be ~10-20%% off)"
                )

        self._init_fallback()

    def _init_azure_tokenizer(self, model_name: str) -> None:
        """Use tiktoken with the model's native encoding."""
        try:
            self._tokenizer = tiktoken.encoding_for_model(model_name)
            self._mode = "tiktoken"
            logger.info(f"Loaded tiktoken encoding for Azure model '{model_name}'")
        except KeyError:
            logger.warning(
                f"No tiktoken encoding found for '{model_name}', "
                f"falling back to cl100k_base"
            )
            self._init_fallback()

    def _init_fallback(self) -> None:
        """Fall back to tiktoken cl100k_base — works for estimation."""
        self._tokenizer = tiktoken.get_encoding("cl100k_base")
        self._mode = "tiktoken_fallback"
        logger.info(
            f"Using tiktoken cl100k_base fallback for '{self._model_name}' — "
            f"token counts are approximate"
        )

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.
        
        Returns:
            Token count. Accuracy depends on mode:
              - hf: ±1 token (exact model tokenizer)
              - tiktoken: exact for OpenAI models
              - tiktoken_fallback: ~10-20% off for non-OpenAI models
        """
        if self._mode == "hf":
            return len(self._tokenizer.encode(text))
        else:
            return len(self._tokenizer.encode(text))

    def count_messages_tokens(self, messages: list[dict]) -> int:
        """
        Count tokens across a list of chat messages.
        
        Each message is expected to have 'role' and 'content' keys.
        Adds ~4 tokens per message for chat formatting overhead.
        """
        total = 0
        for msg in messages:
            # ~4 tokens per message for role/formatting overhead
            total += 4
            total += self.count_tokens(msg.get("content", ""))
            total += self.count_tokens(msg.get("role", ""))
        # Every reply is primed with <|start|>assistant<|message|>
        total += 3
        return total

    @property
    def mode(self) -> str:
        """Return the tokenizer mode: 'hf', 'tiktoken', or 'tiktoken_fallback'."""
        return self._mode
