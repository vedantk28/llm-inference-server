"""
Token budget enforcement middleware.

Checks that every incoming chat completion request fits within the configured
token budget (MAX_TOKENS_BUDGET). If the estimated tokens (input + requested
output) exceed the budget, the request is rejected with HTTP 400.

Why HTTP 400 (not 429)?
  - 429 "Too Many Requests" is for rate limiting (RFC 6585) — it means
    "you're sending requests too fast, try again later"
  - Token budget violations are a request validation error — the request
    itself is asking for more tokens than allowed. It would fail no matter
    how slowly you send it.
  - Rate limiting (if needed later) would be a separate middleware using 429.

Headers added to every response:
  - X-Tokens-Used: tokens consumed by this request (input + output)
  - X-Tokens-Remaining: budget remaining after this request
  - X-Tokens-Budget: total configured budget
"""

import logging

from app.config import settings
from app.utils.tokenizer import HybridTokenizer

logger = logging.getLogger(__name__)


class TokenBudgetChecker:
    """
    Validates that requests fit within the configured token budget.
    
    Uses the HybridTokenizer to count input tokens, then checks
    that input_tokens + max_tokens <= MAX_TOKENS_BUDGET.
    """

    def __init__(self, tokenizer: HybridTokenizer):
        self._tokenizer = tokenizer
        self._budget = settings.max_tokens_budget

    def check(
        self, messages: list[dict], max_tokens: int
    ) -> tuple[bool, int, str]:
        """
        Check if the request fits within the token budget.
        
        Args:
            messages: Chat messages to estimate input tokens for
            max_tokens: Requested max completion tokens
            
        Returns:
            Tuple of (is_within_budget, input_token_count, error_message)
            error_message is empty if within budget.
        """
        input_tokens = self._tokenizer.count_messages_tokens(messages)
        total_requested = input_tokens + max_tokens

        if total_requested > self._budget:
            error_msg = (
                f"Token budget exceeded: {total_requested} tokens requested "
                f"({input_tokens} input + {max_tokens} max output) "
                f"exceeds budget of {self._budget}"
            )
            logger.info(error_msg)
            return False, input_tokens, error_msg

        return True, input_tokens, ""

    @property
    def budget(self) -> int:
        """Return the configured token budget."""
        return self._budget
