"""Instruction variation augmentation: use Claude API to paraphrase task descriptions."""

import os

import anthropic
import numpy as np

from augmentations import register
from augmentations.base import Augmentation
import config


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
        """Generate paraphrases for a single task description using Claude."""
        client = self._get_client()
        prompt = config.LLM_PARAPHRASE_PROMPT.format(
            num_paraphrases=self.num_paraphrases,
            task=task,
        )
        try:
            response = client.messages.create(
                model=config.LLM_MODEL,
                max_tokens=config.LLM_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.BadRequestError as e:
            raise RuntimeError(
                f"Claude API error: {e}. Check your API credits at https://console.anthropic.com/"
            ) from e
        lines = [line.strip() for line in response.content[0].text.strip().split("\n") if line.strip()]
        return lines[:self.num_paraphrases]

    def prepare(self, tasks: list[str]):
        """Pre-generate paraphrases for all unique tasks. Call before augmentation loop."""
        unique_tasks = set(tasks)
        for task in unique_tasks:
            if task not in self._paraphrases:
                print(f"  Generating paraphrases for: '{task}'")
                self._paraphrases[task] = self._generate_paraphrases(task)
                print(f"  -> {self._paraphrases[task]}")

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
                self._current_paraphrase = original_task

        result["task"] = self._current_paraphrase
        return result
