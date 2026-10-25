import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Hashable
from scipy.spatial.distance import jensenshannon


def period_sort_key(period_label: str) -> tuple[float, str]:
    """Sort period labels by their first four-digit year."""
    match = re.search(r"\d{4}", period_label)

    if match is None:
        return math.inf, period_label

    return int(match.group()), period_label

def load_label_counts(input_path: Path, word2period: dict) -> dict[str, dict[str, Counter]]:
    """
    Return fractional label counts indexed by word and period.

    Structure:
        counts[word][period][label] = fractional count

    For example, labels [3, 4] contribute:
        0.5 to label 3
        0.5 to label 4
    """
    counts: dict[str, dict[str, Counter]] = defaultdict(
        lambda: defaultdict(Counter)
    )

    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                observation = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

            missing_fields = {
                field
                for field in ("word", "sentence_id", "label")
                if field not in observation
            }

            if missing_fields:
                raise ValueError(
                    f"Missing field(s) on line {line_number}: "
                    f"{', '.join(sorted(missing_fields))}"
                )

            word = observation["word"]
            sentence_id = observation["sentence_id"]
            labels = observation["label"]
            period = word2period[word][sentence_id]

            if not isinstance(word, str) or not word:
                raise ValueError(
                    f"'word' must be a non-empty string on line {line_number}"
                )

            if not isinstance(labels, list):
                raise ValueError(
                    f"'label' must be a list on line {line_number}"
                )

            weight = 1.0 / len(labels)

            for label in labels:
                counts[word][period][label] += weight

    return {word: dict(period_counts) for word, period_counts in counts.items()}


def load_gold_label_counts(input_path: Path) -> dict[str, dict[str, Counter]]:
    """
    Return fractional label counts indexed by word and period.

    Structure:
        counts[word][period][label] = fractional count

    For example, labels [3, 4] contribute:
        0.5 to label 3
        0.5 to label 4
    """
    word2period = {}
    counts: dict[str, dict[str, Counter]] = defaultdict(
        lambda: defaultdict(Counter)
    )

    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                observation = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

            missing_fields = {
                field
                for field in ("word", "period_label", "sentence_id", "label")
                if field not in observation
            }

            if missing_fields:
                raise ValueError(
                    f"Missing field(s) on line {line_number}: "
                    f"{', '.join(sorted(missing_fields))}"
                )

            word = observation["word"]
            period = observation["period_label"]
            sentence_id = observation["sentence_id"]
            labels = observation["label"]
            if not word in word2period:
                word2period[word] = {}
            word2period[word][sentence_id] = period

            if not isinstance(word, str) or not word:
                raise ValueError(
                    f"'word' must be a non-empty string on line {line_number}"
                )

            if not isinstance(period, str) or not period:
                raise ValueError(
                    f"'period_label' must be a non-empty string "
                    f"on line {line_number}"
                )

            if not isinstance(labels, list):
                raise ValueError(
                    f"'label' must be a list on line {line_number}"
                )

            weight = 1.0 / len(labels)

            for label in labels:
                counts[word][period][label] += weight

    return {word: dict(period_counts) for word, period_counts in counts.items()}, word2period


def normalize_distribution(counts: Counter, labels: list[Hashable]) -> list[float]:
    """Convert fractional label counts into probabilities."""
    total = sum(counts.values())

    if total <= 0:
        raise ValueError("Cannot normalize an empty distribution")

    return [counts.get(label, 0.0) / total for label in labels]



def calculate_JSD_distances(counts: dict[str, dict[str, Counter]]) -> list[dict]:

    results = {}

    for word in sorted(counts):
        period_counts = counts[word]
        ordered_periods = sorted(period_counts, key=period_sort_key)

        if len(ordered_periods) < 2:
            raise Exception(f'{word} has less than 2 time periods.')

        for previous_period, current_period in zip(ordered_periods,ordered_periods[1:]):
            previous_counts = period_counts[previous_period]
            current_counts = period_counts[current_period]

            # Use the union of labels found in the two periods.
            labels = sorted(
                set(previous_counts) | set(current_counts),
                key=lambda value: (str(type(value)), str(value)),
            )

            previous_distribution = normalize_distribution(
                previous_counts,
                labels,
            )
            current_distribution = normalize_distribution(
                current_counts,
                labels,
            )

            distance = jensenshannon(
                previous_distribution,
                current_distribution,
                base=2.0,
            )

            results[f'{word}_{previous_period}_{current_period}'] = distance

    return results

def load_subtask2_labels(input_path: Path) -> dict[str, dict[str, Counter]]:

    results = {}

    with open(input_path) as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                observation = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

            missing_fields = {
                field
                for field in ("word", "label")
                if field not in observation
            }

            if missing_fields:
                raise ValueError(
                    f"Missing field(s) on line {line_number}: "
                    f"{', '.join(sorted(missing_fields))}"
                )

            word = observation["word"]
            sentence_id = observation["sentence_id"]
            label = observation["label"]

            if not type(label) == int:
                raise ValueError(
                    f"Label must be an integer on line {line_number}: "
                )

            if not label == 0 and not label == 1:
                raise ValueError(
                    f"Label must be 0 or 1 on line {line_number}: "
                )

            results[f'{word};{sentence_id}'] = label

    return results



def load_subtask1_labels(input_path: Path) -> dict[str, dict[str, Counter]]:

    results = {}

    with open(input_path) as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                observation = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

            missing_fields = {
                field
                for field in ("word", "label")
                if field not in observation
            }

            if missing_fields:
                raise ValueError(
                    f"Missing field(s) on line {line_number}: "
                    f"{', '.join(sorted(missing_fields))}"
                )

            word = observation["word"]
            sentence_id = observation["sentence_id"]
            label = observation["label"]

            if not type(label) == list:
                raise ValueError(
                    f"Label must be a list on line {line_number}: "
                )

            results[f'{word};{sentence_id}'] = label

    return results