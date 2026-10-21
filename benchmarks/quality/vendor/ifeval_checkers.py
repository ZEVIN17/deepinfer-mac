# ruff: noqa
# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Unmodified selected Google IFEval checker classes; thin base stub below.

Source retrieved 2026-10-21; original source SHA256: 60e086f5342a03ce8e18b64bbcccf86308f523c08aa826707a562150a52f3edf
https://github.com/google-research/google-research/blob/master/instruction_following_eval/instructions.py
Only explicit dataset kwargs are used, so random/default argument paths are unused.
"""
import json
import random
import re

_NUM_PLACEHOLDERS = 4
_NUM_BULLETS = 5
_NUM_KEYWORDS = 2
_ENDING_OPTIONS = ()

class Instruction:
    """Base interface only; selected original subclasses implement scoring."""

class PlaceholderChecker(Instruction):
  """Check the placeholders in template writing."""

  def build_description(self, *, num_placeholders = None):  # pyrefly: ignore[bad-override]
    """Build the instruction description.

    Args:
      num_placeholders: An integer denoting the minimum number of
        placeholders required in the response.

    Returns:
      A string representing the instruction description.
    """
    self._num_placeholders = num_placeholders
    if self._num_placeholders is None or self._num_placeholders < 0:
      self._num_placeholders = random.randint(1, _NUM_PLACEHOLDERS)
    self._description_pattern = (
        "The response must contain at least {num_placeholders} placeholders " +
        "represented by square brackets, such as [address].")
    return self._description_pattern.format(
        num_placeholders=self._num_placeholders)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"num_placeholders": self._num_placeholders}  # pyrefly: ignore[bad-return]

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["num_placeholders"]

  def check_following(self, value):
    """Check if the number of placeholders follows the instruction.

    Args:
      value: A string representing the response.

    Returns:
      True if the actual number of placeholders in the response is greater than
      or equal to `num_placeholders`; otherwise, False.
    """
    placeholders = re.findall(r"\[.*?\]", value)
    num_placeholders = len(placeholders)
    return num_placeholders >= self._num_placeholders


class BulletListChecker(Instruction):
  """Checks the bullet list in the prompt."""

  def build_description(self, *, num_bullets = None):  # pyrefly: ignore[bad-override]
    """Build the instruction description.

    Args:
      num_bullets: An integer specifying the exact number of bullet lists
        that is required to appear in the response.

    Returns:
      A string representing the instruction description.
    """
    self._num_bullets = num_bullets
    if self._num_bullets is None or self._num_bullets < 0:
      self._num_bullets = random.randint(1, _NUM_BULLETS)
    self._description_pattern = (
        "Your answer must contain exactly {num_bullets} bullet points. " +
        "Use the markdown bullet points such as:\n" +
        "* This is point 1. \n" +
        "* This is point 2")
    return self._description_pattern.format(
        num_bullets=self._num_bullets)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"num_bullets": self._num_bullets}  # pyrefly: ignore[bad-return]

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["num_bullets"]

  def check_following(self, value):
    r"""Check if the number of bullet lists meets the requirement.

    Args:
      value: A string representing the response. The response is expected to
        contain some bullet lists that start with `\*`.

    Returns:
      True if the actual number of bullet lists in the response meets the
      requirement.
    """
    bullet_lists = re.findall(r"^\s*\*[^\*].*$", value, flags=re.MULTILINE)
    bullet_lists_2 = re.findall(r"^\s*-.*$", value, flags=re.MULTILINE)
    num_bullet_lists = len(bullet_lists) + len(bullet_lists_2)
    return num_bullet_lists == self._num_bullets


class KeywordChecker(Instruction):
  """Check the exisitence of certain keywords."""

  def build_description(self, *, keywords = None  # pyrefly: ignore[bad-override]
                        ):
    """Build the instruction description.

    Args:
      keywords: A sequence of strings representing the keywords that are
        expected in the response.

    Returns:
      A string representing the instruction description.
    """

    if not keywords:
      self._keywords = instructions_util.generate_keywords(
          num_keywords=_NUM_KEYWORDS)
    else:
      self._keywords = keywords
    self._keywords = sorted(self._keywords)

    self._description_pattern = ("Include keywords {keywords} in the response.")

    return self._description_pattern.format(keywords=self._keywords)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"keywords": self._keywords}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["keywords"]

  def check_following(self, value):
    """Check if the response contain the expected keywords."""
    for keyword in self._keywords:
      if not re.search(keyword, value, flags=re.IGNORECASE):
        return False
    return True


class JsonFormat(Instruction):
  """Check the Json format."""

  def build_description(self):  # pyrefly: ignore[bad-override]
    self._description_pattern = (
        "Entire output should be wrapped in JSON format. You can use markdown"
        " ticks such as ```."
    )
    return self._description_pattern

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return None

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return []

  def check_following(self, value):
    value = (
        value.strip()
        .removeprefix("```json")
        .removeprefix("```Json")
        .removeprefix("```JSON")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
      json.loads(value)
    except ValueError as _:
      return False
    return True


class EndChecker(Instruction):
  """Checks that the prompt ends with a given phrase."""

  def build_description(self, *, end_phrase = None):  # pyrefly: ignore[bad-override]
    """Build the instruction description.

    Args:
      end_phrase: A string representing the phrase the response should end with.

    Returns:
      A string representing the instruction description.
    """
    self._end_phrase = (
        end_phrase.strip() if isinstance(end_phrase, str) else end_phrase
    )
    if self._end_phrase is None:
      self._end_phrase = random.choice(_ENDING_OPTIONS)
    self._description_pattern = (
        "Finish your response with this exact phrase {ender}. "
        "No other words should follow this phrase.")
    return self._description_pattern.format(ender=self._end_phrase)

  def get_instruction_args(self):
    return {"end_phrase": self._end_phrase}  # pyrefly: ignore[bad-return]

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["end_phrase"]

  def check_following(self, value):
    """Checks if the response ends with the expected phrase."""
    value = value.strip().strip("\"").lower()
    self._end_phrase = self._end_phrase.strip().lower()  # pyrefly: ignore[missing-attribute]
    return value.endswith(self._end_phrase)


class TitleChecker(Instruction):
  """Checks the response for a title."""

  def build_description(self):  # pyrefly: ignore[bad-override]
    """Build the instruction description."""
    self._description_pattern = (
        "Your answer must contain a title, wrapped in double angular brackets,"
        " such as <<poem of joy>>."
    )
    return self._description_pattern

  def get_instruction_args(self):
    return None

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return []

  def check_following(self, value):
    """Checks if the response contains a title."""
    pattern = r"<<[^\n]+>>"
    re_pattern = re.compile(pattern)
    titles = re.findall(re_pattern, value)

    for title in titles:
      if title.lstrip("<").rstrip(">").strip():
        return True
    return False


class CommaChecker(Instruction):
  """Checks the response for no commas."""

  def build_description(self):  # pyrefly: ignore[bad-override]
    """Build the instruction description."""
    self._description_pattern = (
        "In your entire response, refrain from the use of any commas."
    )
    return self._description_pattern

  def get_instruction_args(self):
    return None

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return []

  def check_following(self, value):
    """Checks that the response does not contain commas."""
    return not re.search(r"\,", value)
