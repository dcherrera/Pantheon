"""
gpqa_dataset.py — GPQA Diamond rows, deterministic per-seed choice shuffling, and the prompt.

The prompt is OpenAI simple-evals' QUERY_TEMPLATE_MULTICHOICE (the de-facto GPQA zero-shot CoT
prompt; tt-hous and Ornith AI publish no template). The shuffle is keyed on (seed, Record ID),
not on row order, so a subset or a resumed run sees the same permutation as the full run.
Question text never leaves this module except inside the request body.

QUERY_TEMPLATE_MULTICHOICE is from OpenAI simple-evals (https://github.com/openai/simple-evals),
Copyright (c) 2024 OpenAI, MIT License (LICENSES/MIT.txt; see NOTICE).

Status: works.
Issues: None
Todo: None
"""
import csv
import hashlib
import os
import random

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(EVAL_DIR, "data")
DIAMOND_CSV = os.path.join(DATA_DIR, "gpqa_diamond.csv")
CHOICE_LETTERS = "ABCD"

QUERY_TEMPLATE_MULTICHOICE = (
    "Answer the following multiple choice question. The last line of your response should be "
    "of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. "
    "Think step by step before answering.\n\n"
    "{question}\n\n"
    "A) {A}\nB) {B}\nC) {C}\nD) {D}"
)


def load_diamond_questions(csv_path):
    """List of dicts {record_id, question, correct, incorrect[3]} in file order."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        {
            "record_id": row["Record ID"].strip(),
            "question": row["Question"].strip(),
            "correct": row["Correct Answer"].strip(),
            "incorrect": [row[f"Incorrect Answer {i}"].strip() for i in (1, 2, 3)],
        }
        for row in rows
    ]


def stable_int_from_text(text):
    """A 63-bit integer derived from text with sha256 (stable across Python runs, unlike hash())."""
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big") >> 1


def choice_permutation(seed, record_id):
    """permutation[slot] = index into [correct, incorrect1, incorrect2, incorrect3]."""
    return random.Random(stable_int_from_text(f"gpqa-diamond-shuffle:{seed}:{record_id}")).sample(range(4), 4)


def request_sampling_seed(seed, record_id):
    """Per-request sampler seed sent to the serve, so a single question is reproducible."""
    return stable_int_from_text(f"gpqa-diamond-sample:{seed}:{record_id}") % (1 << 31)


def build_shuffled_item(question_row, seed):
    """{record_id, prompt, correct_letter, permutation} for one question under one seed."""
    options = [question_row["correct"]] + question_row["incorrect"]
    permutation = choice_permutation(seed, question_row["record_id"])
    shuffled = [options[i] for i in permutation]
    prompt = QUERY_TEMPLATE_MULTICHOICE.format(
        question=question_row["question"], A=shuffled[0], B=shuffled[1], C=shuffled[2], D=shuffled[3]
    )
    return {
        "record_id": question_row["record_id"],
        "prompt": prompt,
        "correct_letter": CHOICE_LETTERS[permutation.index(0)],
        "permutation": permutation,
    }
