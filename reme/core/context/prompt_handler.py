"""Module for managing and formatting prompt templates from files or dictionaries.

This module provides a PromptHandler class that:
- Loads prompts from YAML/JSON files or dictionaries
- Supports multi-language prompts with automatic suffix handling
- Provides conditional line filtering using boolean flags
- Formats prompts with template variable substitution
- Validates format strings and provides helpful error messages
"""

import json
from pathlib import Path
from string import Formatter
from typing import Any, Dict, Optional, Union

import yaml
from loguru import logger

from .base_context import BaseContext


class PromptNotFoundError(KeyError):
    """Exception raised when a requested prompt template is not found."""

    def __init__(self, prompt_name: str, available_prompts: list[str]):
        self.prompt_name = prompt_name
        self.available_prompts = available_prompts
        super().__init__(
            f"Prompt '{prompt_name}' not found. "
            f"Available prompts: {', '.join(available_prompts[:10])}"
            f"{'...' if len(available_prompts) > 10 else ''}",
        )


class PromptFormattingError(ValueError):
    """Exception raised when prompt formatting fails."""


class PromptHandler(BaseContext):
    """A context-aware handler for loading, retrieving, and formatting prompt templates.

    This handler supports:
    - Loading prompts from YAML/JSON files or dictionaries
    - Multi-language prompt support with automatic language suffix
    - Conditional line filtering using boolean flags (e.g., [debug], [verbose])
    - Template variable substitution with validation
    - Method chaining for fluent API

    Examples:
        >>> handler = PromptHandler(language="en")
        >>> handler.load_prompt_dict({
        ...     "greeting_en": "Hello, {name}!",
        ...     "farewell_en": "[debug]Debug mode\\nGoodbye, {name}!"
        ... })
        >>> handler.prompt_format("greeting", name="Alice")
        'Hello, Alice!'
        >>> handler.prompt_format("farewell", name="Bob", debug=False)
        'Goodbye, Bob!'
    """

    def __init__(self, language: str = "", **kwargs):
        """Initialize the PromptHandler with optional language configuration.

        Args:
            language: Language code to append as suffix (e.g., "en", "zh", "ja").
                     If provided, get_prompt will automatically try to find
                     prompts with this suffix (e.g., "greeting" -> "greeting_en").
            **kwargs: Additional key-value pairs to initialize the context.
        """
        super().__init__(**kwargs)
        self.language: str = language.strip()

    def load_prompt_by_file(
        self,
        prompt_file_path: Optional[Union[Path, str]] = None,
        overwrite: bool = True,
    ) -> "PromptHandler":
        """Load prompt configurations from a YAML or JSON file into the context.

        Supports both YAML (.yaml, .yml) and JSON (.json) file formats.
        Non-existent files are silently skipped.

        Args:
            prompt_file_path: Path to the prompt configuration file.
                            If None, returns self without changes.
            overwrite: If True, allows overwriting existing prompts with warnings.
                      If False, skips existing prompts without overwriting.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If file format is not supported.
            yaml.YAMLError: If YAML parsing fails.
            json.JSONDecodeError: If JSON parsing fails.
        """
        if prompt_file_path is None:
            return self

        if isinstance(prompt_file_path, str):
            prompt_file_path = Path(prompt_file_path)

        if not prompt_file_path.exists():
            logger.warning(f"Prompt file not found: {prompt_file_path}")
            return self

        suffix = prompt_file_path.suffix.lower()

        try:
            with prompt_file_path.open(encoding="utf-8") as f:
                if suffix in [".yaml", ".yml"]:
                    prompt_dict = yaml.safe_load(f)
                elif suffix == ".json":
                    prompt_dict = json.load(f)
                else:
                    raise ValueError(
                        f"Unsupported file format: {suffix}. " f"Supported formats: .yaml, .yml, .json",
                    )

            logger.info(f"Loaded {len(prompt_dict or {})} prompts from {prompt_file_path}")
            self.load_prompt_dict(prompt_dict, overwrite=overwrite)

        except (yaml.YAMLError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse prompt file {prompt_file_path}: {e}")
            raise

        return self

    def load_prompt_dict(
        self,
        prompt_dict: Optional[Dict[str, Any]] = None,
        overwrite: bool = True,
    ) -> "PromptHandler":
        """Merge a dictionary of prompt strings into the current context.

        Only string values are stored as prompts. Non-string values are skipped.

        Args:
            prompt_dict: Dictionary mapping prompt names to prompt template strings.
            overwrite: If True, allows overwriting existing prompts with warnings.
                      If False, skips existing prompts without overwriting.

        Returns:
            Self for method chaining.
        """
        if not prompt_dict:
            return self

        for key, value in prompt_dict.items():
            if not isinstance(value, str):
                logger.debug(f"Skipping non-string prompt: key={key}, type={type(value)}")
                continue

            if key in self:
                if overwrite:
                    logger.warning(
                        f"Overwriting prompt '{key}': " f"old length={len(self[key])}, new length={len(value)}",
                    )
                    self[key] = value
                else:
                    logger.debug(f"Skipping existing prompt: key={key}")
            else:
                logger.debug(f"Adding new prompt: key={key}, length={len(value)}")
                self[key] = value

        return self

    def get_prompt(self, prompt_name: str, fallback_to_base: bool = True) -> str:
        """Retrieve a prompt by name with automatic language suffix handling.

        If a language is configured, this method will:
        1. First try to find the prompt with language suffix (e.g., "greeting_en")
        2. If not found and fallback_to_base is True, try the base name (e.g., "greeting")
        3. Otherwise, raise PromptNotFoundError

        Args:
            prompt_name: Name of the prompt to retrieve.
            fallback_to_base: If True and language-specific prompt not found,
                            fallback to prompt without language suffix.

        Returns:
            The prompt template string, stripped of leading/trailing whitespace.

        Raises:
            PromptNotFoundError: If the prompt is not found.
        """
        # Try with language suffix first
        if self.language and not prompt_name.endswith(f"_{self.language}"):
            key_with_lang = f"{prompt_name}_{self.language}"
            if key_with_lang in self:
                return self[key_with_lang].strip()

        # Try base name
        if prompt_name in self:
            return self[prompt_name].strip()

        # Try fallback if enabled
        if fallback_to_base and self.language:
            # Check if prompt_name already has language suffix, try without it
            if prompt_name.endswith(f"_{self.language}"):
                base_name = prompt_name[: -(len(self.language) + 1)]
                if base_name in self:
                    return self[base_name].strip()

        # Not found, raise error with helpful message
        available = list(self.keys())
        raise PromptNotFoundError(prompt_name, available)

    def has_prompt(self, prompt_name: str) -> bool:
        """Check if a prompt exists (with or without language suffix).

        Args:
            prompt_name: Name of the prompt to check.

        Returns:
            True if the prompt exists, False otherwise.
        """
        try:
            self.get_prompt(prompt_name)
            return True
        except PromptNotFoundError:
            return False

    def list_prompts(self, language_filter: Optional[str] = None) -> list[str]:
        """List all available prompt names.

        Args:
            language_filter: If provided, only return prompts for this language.
                           If None, return all prompts.

        Returns:
            List of prompt names.
        """
        if language_filter is None:
            return list(self.keys())

        suffix = f"_{language_filter.strip()}"
        return [key for key in self.keys() if key.endswith(suffix)]

    @staticmethod
    def _extract_format_fields(template: str) -> set[str]:
        """Extract all format field names from a template string.

        Args:
            template: Template string with {variable} placeholders.

        Returns:
            Set of field names used in the template.
        """
        return {field_name for _, field_name, _, _ in Formatter().parse(template) if field_name is not None}

    @staticmethod
    def _filter_conditional_lines(prompt: str, flags: Dict[str, bool]) -> str:
        """Filter lines based on boolean flags.

        Lines starting with [flag_name] are conditionally included based on
        the value of flags[flag_name]. If True, the line is included (without
        the flag marker). If False, the line is excluded.

        Args:
            prompt: The prompt text with conditional markers.
            flags: Dictionary of flag names to boolean values.

        Returns:
            Filtered prompt text.
        """
        filtered_lines = []

        for line in prompt.split("\n"):
            # Check each flag
            matched_flag = None
            for flag_name in flags:
                marker = f"[{flag_name}]"
                if line.startswith(marker):
                    matched_flag = flag_name
                    break

            if matched_flag is None:
                # No flag marker, always include
                filtered_lines.append(line)
            elif flags[matched_flag]:
                # Flag is True, include without marker
                marker = f"[{matched_flag}]"
                filtered_lines.append(line[len(marker) :])
            # else: Flag is False, skip this line

        return "\n".join(filtered_lines)

    def prompt_format(
        self,
        prompt_name: str,
        validate: bool = True,
        **kwargs,
    ) -> str:
        """Format a prompt with conditional line filtering and variable substitution.

        This method performs two-stage formatting:
        1. Conditional line filtering: Lines marked with [flag] are included only
           if the corresponding boolean kwarg is True.
        2. Variable substitution: Template variables {var} are replaced with
           provided values.

        Args:
            prompt_name: Name of the prompt to format.
            validate: If True, check that all required template variables are provided.
            **kwargs: Keyword arguments for formatting. Boolean values are treated as
                     conditional flags, other values are used for template substitution.

        Returns:
            Formatted prompt string.

        Raises:
            PromptNotFoundError: If the prompt is not found.
            PromptFormattingError: If validation fails or formatting errors occur.

        Examples:
            >>> handler = PromptHandler()
            >>> handler["test"] = "[debug]Debug: {info}\\nResult: {value}"
            >>> handler.prompt_format("test", debug=False, info="test", value=42)
            'Result: 42'
            >>> handler.prompt_format("test", debug=True, info="test", value=42)
            'Debug: test\\nResult: 42'
        """
        # Get the prompt template
        prompt = self.get_prompt(prompt_name)

        # Separate boolean flags from format variables
        flag_kwargs = {k: v for k, v in kwargs.items() if isinstance(v, bool)}
        format_kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, bool)}

        # Step 1: Filter conditional lines
        if flag_kwargs:
            prompt = self._filter_conditional_lines(prompt, flag_kwargs)

        # Step 2: Validate required fields if requested
        if validate:
            required_fields = self._extract_format_fields(prompt)
            missing_fields = required_fields - set(format_kwargs.keys())

            if missing_fields:
                raise PromptFormattingError(
                    f"Missing required format variables for prompt '{prompt_name}': "
                    f"{', '.join(sorted(missing_fields))}",
                )

        # Step 3: Format with variables
        try:
            if format_kwargs:
                prompt = prompt.format(**format_kwargs)
        except KeyError as e:
            raise PromptFormattingError(
                f"Format error in prompt '{prompt_name}': missing variable {e}",
            ) from e
        except (ValueError, IndexError) as e:
            raise PromptFormattingError(
                f"Format error in prompt '{prompt_name}': {e}",
            ) from e

        return prompt.strip()

    def __repr__(self) -> str:
        """Return a string representation of the PromptHandler."""
        return f"PromptHandler(language='{self.language}', " f"num_prompts={len(self)})"
