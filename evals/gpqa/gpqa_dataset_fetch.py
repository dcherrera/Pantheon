#!/usr/bin/env python3
"""
gpqa_dataset_fetch.py — download GPQA Diamond (Idavidrein/gpqa, gated) into evals/gpqa/data/.

Uses the Mac's existing Hugging Face login (HF_TOKEN env, else ~/.cache/huggingface/token).
The token is read, sent as an Authorization header, and never printed or logged. The dataset
asks that its questions not be revealed online: data/ is gitignored and nothing here prints
question text — only the row count and the file's sha256.

  python3 evals/gpqa/gpqa_dataset_fetch.py [--force]

Status: works (stdlib only).
Issues: None
Todo: None
"""
import argparse
import hashlib
import os
import sys
import urllib.error
import urllib.request

from gpqa_dataset import DATA_DIR, DIAMOND_CSV, load_diamond_questions

DATASET_REPO = "Idavidrein/gpqa"
DATASET_FILE = "gpqa_diamond.csv"
DATASET_REVISION = "main"


def read_hf_token():
    """The HF token from HF_TOKEN or the hub's token file; None if neither exists."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    path = os.path.expanduser(os.environ.get("HF_TOKEN_PATH", "~/.cache/huggingface/token"))
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip() or None
    except FileNotFoundError:
        return None


def download_gated_file(repo, filename, revision, token, out_path):
    """Fetch one file of a gated HF dataset to out_path (atomic rename). Raises on HTTP error."""
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    partial_path = out_path + ".partial"
    with urllib.request.urlopen(request, timeout=120) as response, open(partial_path, "wb") as f:
        while True:
            chunk = response.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
    os.replace(partial_path, out_path)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    if args.force or not os.path.exists(DIAMOND_CSV):
        token = read_hf_token()
        if not token:
            sys.exit("gpqa_dataset_fetch: no HF login (set HF_TOKEN or run `hf auth login`)")
        try:
            download_gated_file(DATASET_REPO, DATASET_FILE, DATASET_REVISION, token, DIAMOND_CSV)
        except urllib.error.HTTPError as e:
            hint = " (accept the dataset terms on its HF page first)" if e.code in (401, 403) else ""
            sys.exit(f"gpqa_dataset_fetch: HTTP {e.code} fetching {DATASET_REPO}/{DATASET_FILE}{hint}")

    questions = load_diamond_questions(DIAMOND_CSV)
    print(f"gpqa_dataset_fetch: {len(questions)} questions, sha256 {file_sha256(DIAMOND_CSV)}")
    if len(questions) != 198:
        sys.exit("gpqa_dataset_fetch: expected 198 Diamond questions")


if __name__ == "__main__":
    main()
