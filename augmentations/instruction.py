"""Instruction variation augmentation: use Claude API to paraphrase task descriptions.

API resilience: Uses exponential backoff retry (3 attempts) for transient errors
(rate limits, 502s, timeouts). On persistent failure, falls back to the original
task text rather than crashing the pipeline — losing instruction variation on one
task is acceptable, losing an hour of video encoding is not.
"""

import os
import time

import anthropic
import numpy as np

from augmentations import register
from augmentations.base import Augmentation
import config

MAX_RETRIES = 3
INITIAL_BACKOFF_S = 2.0


@register("instruction")
class InstructionAugmentation(Augmentation):
    """Rewrite task descriptions using Claude API to increase language variation.

    At init, generates N paraphrases for each unique task. During augmentation,
    each episode gets a randomly selected paraphrase.
    """

    def __init__(self, num_paraphrases: int = config.INSTRUCTION_NUM_PARAPHRASES):
        self.num_paraphrases = num_paraphrases
        self._paraphrases: dict[str, list[str]] = {}
        self._current_paraphrase: str | None = None
        self._client = None

    @property
    def name(self) -> str:
        return "instruction"

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY environment variable required for instruction augmentation. "
                    "Copy .env.example to .env and add your key."
                )
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def _generate_paraphrases(self, task: str) -> list[str]:
        """Generate paraphrases with retry logic and graceful fallback.

        Retries up to MAX_RETRIES times with exponential backoff for transient
        errors (rate limits, server errors, timeouts). On persistent failure,
        returns empty list (caller falls back to original task text).
        """
        client = self._get_client()
        prompt = config.LLM_PARAPHRASE_PROMPT.format(
            num_paraphrases=self.num_paraphrases,
            task=task,
        )

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = client.messages.create(
                    model=config.LLM_MODEL,
                    max_tokens=config.LLM_MAX_TOKENS,
                    messages=[{"role": "user", "content": prompt}],
                )
                lines = [
                    line.strip()
                    for line in response.content[0].text.strip().split("\n")
                    if line.strip()
                ]
                return lines[:self.num_paraphrases]

            except anthropic.BadRequestError as e:
                # Non-retryable (e.g. insufficient credits) — fail immediately
                print(f"  WARNING: Claude API non-retryable error: {e}")
                print(f"  Falling back to original task text for: '{task}'")
                return []

            except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as e:
                last_error = e
                backoff = INITIAL_BACKOFF_S * (2 ** attempt)
                print(f"  WARNING: Claude API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                print(f"  Retrying in {backoff:.1f}s...")
                time.sleep(backoff)

            except Exception as e:
                # Unexpected error — log and fall back
                print(f"  WARNING: Unexpected error calling Claude API: {e}")
                print(f"  Falling back to original task text for: '{task}'")
                return []

        # All retries exhausted
        print(f"  WARNING: All {MAX_RETRIES} retries failed for: '{task}' (last error: {last_error})")
        print(f"  Falling back to original task text.")
        return []

    def prepare(self, tasks: list[str], robot_cfg: dict | None = None, stats: dict | None = None):
        """Pre-generate paraphrases for all unique tasks. Call before augmentation loop."""
        unique_tasks = set(tasks)
        for task in unique_tasks:
            if task not in self._paraphrases:
                print(f"  Generating paraphrases for: '{task}'")
                paraphrases = self._generate_paraphrases(task)
                self._paraphrases[task] = paraphrases
                if paraphrases:
                    print(f"  -> {paraphrases}")
                else:
                    print(f"  -> (using original text as fallback)")

    def on_episode_start(self):
        """Select a random paraphrase for this episode."""
        self._current_paraphrase = None  # Will be set on first frame

    def apply_frame(self, frame_dict: dict, metadata: dict) -> dict:
        result = dict(frame_dict)
        original_task = result.get("task", "")

        if self._current_paraphrase is None:
            paraphrases = self._paraphrases.get(original_task, [])
            if paraphrases:
                self._current_paraphrase = paraphrases[np.random.randint(len(paraphrases))]
            else:
                # Fallback: use original task text (API failed or no paraphrases generated)
                self._current_paraphrase = original_task

        result["task"] = self._current_paraphrase
        return result
