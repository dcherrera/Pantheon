"""
gpqa_answer_extraction.py — pull the chosen letter (A-D) out of a model's final answer.

Order of patterns, first that matches wins; within a pattern the LAST match wins (a model that
reconsiders states its final choice last):
  1. answer_line   simple-evals' own regex  (?i)Answer[ \\t]*:[ \\t]*\\$?([A-D])\\$?  ,
                   widened to tolerate markdown/brackets around the letter (**Answer:** (B)),
                   with the letter kept case-sensitive so "Answer: a mix..." is not an A.
  2. boxed         \\boxed{B} / \\boxed{\\text{B}}
  3. answer_is     "the answer is B" / "correct option is (B)"
  4. lone_letter   the whole final line is just a letter: "B", "(B)", "**B**", "B)".
Extraction is from `content` only (the post-</think> answer). A response whose reasoning never
closed has no content; that is reported as a failure with its own reason, never guessed from
the chain of thought. Any leaked <think>...</think> is stripped first.

Status: works; covered by test_gpqa_answer_extraction.py.
Issues: None
Todo: None
"""
import re

_DECOR = r"[\s\*_`\(\[\{\$]*"
_LETTER = r"([A-D])"
_END = r"(?![A-Za-z0-9])"

ANSWER_PATTERNS = (
    ("answer_line", re.compile(r"\**(?i:answer)\**[ \t]*\**[ \t]*:[ \t]*\**" + _DECOR + _LETTER + _END)),
    ("boxed", re.compile(r"\\boxed\{\s*(?:\\text(?:bf)?\{\s*)?\(?" + _LETTER + r"\)?\s*\}?\s*\}")),
    ("answer_is", re.compile(r"(?i:answer|option|choice)\s+is\s*:?" + _DECOR + _LETTER + _END)),
)
LONE_LETTER_LINE = re.compile(r"^[\s\*_`\(\[]*([A-D])[\s\*_`\)\]\.:]*$")
THINK_BLOCK = re.compile(r"(?s)<think>.*?</think>")


def strip_leaked_reasoning(content):
    """Drop a <think> block the serve failed to separate (and anything before a lone </think>)."""
    content = THINK_BLOCK.sub("", content)
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[1]
    return content


def extract_answer_letter(content, finish_reason=None):
    """(letter or None, method) — method names the pattern, or the failure reason when None."""
    text = strip_leaked_reasoning(content or "").strip()
    if not text:
        return None, "truncated_no_answer" if finish_reason == "length" else "empty_content"
    for method, pattern in ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return matches[-1], method
    last_line = text.splitlines()[-1]
    lone = LONE_LETTER_LINE.match(last_line)
    if lone:
        return lone.group(1), "lone_letter"
    return None, "truncated_no_match" if finish_reason == "length" else "no_match"
