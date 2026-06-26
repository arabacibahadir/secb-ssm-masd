from __future__ import annotations

import random
from collections import defaultdict


def build_loso_split(
    subject_map: dict[int, str],
    test_subject: str,
    validation_seed: int,
) -> dict:
    records_by_subject: dict[str, list[int]] = defaultdict(list)
    for record_id, subject_id in subject_map.items():
        records_by_subject[subject_id].append(record_id)
    if test_subject not in records_by_subject:
        raise ValueError(f"unknown test subject: {test_subject}")

    validation_by_subject: dict[str, int] = {}
    for subject_id in sorted(records_by_subject):
        if subject_id == test_subject:
            continue
        choices = sorted(records_by_subject[subject_id])
        rng = random.Random(f"{validation_seed}:{test_subject}:{subject_id}")
        validation_by_subject[subject_id] = rng.choice(choices)

    test_records = sorted(records_by_subject[test_subject])
    validation_records = sorted(validation_by_subject.values())
    excluded = set(test_records) | set(validation_records)
    train_records = sorted(record_id for record_id in subject_map if record_id not in excluded)
    return {
        "test_subject": test_subject,
        "validation_seed": validation_seed,
        "train_records": train_records,
        "validation_records": validation_records,
        "test_records": test_records,
        "validation_by_subject": validation_by_subject,
    }
