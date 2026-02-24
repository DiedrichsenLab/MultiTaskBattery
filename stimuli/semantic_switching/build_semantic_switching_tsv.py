"""
Build semantic_switching.tsv from Excel files in this directory.

- Reads all .xlsx files whose name starts with "selected_" (case-insensitive;
  ignores lock files like ~$Selected_*.xlsx).
- Concatenates them into one table.
- For the sentence column:
  - Strips punctuation from each sentence.
  - Joins words with "|" (word1|word2|word3...).
- Adds a "words" column with the number of words per sentence.
- Writes semantic_switching.tsv in this directory.

Usage (from repo root or this directory):
  python stimuli/semantic_switching/build_semantic_switching_tsv.py
  python build_semantic_switching_tsv.py
"""

import re
from pathlib import Path

import pandas as pd


def _sentence_column_name(df: pd.DataFrame) -> str:
    """Find the sentence column (case-insensitive)."""
    for c in df.columns:
        if str(c).strip().lower() == "sentence":
            return c
    return "sentence"


def _normalize_sentence(raw: str) -> str:
    """Remove punctuation and join words with |."""
    if pd.isna(raw) or not isinstance(raw, str):
        return ""
    # Remove punctuation (keep letters, digits, spaces)
    no_punct = re.sub(r"[^\w\s]", "", raw)
    # Collapse whitespace and strip
    words = no_punct.split()
    return "|".join(words)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    out_path = script_dir / "semantic_switching.tsv"

    # Find Excel files starting with "selected_" (ignore lock files ~$...)
    xlsx_files = sorted(
        f
        for f in script_dir.glob("*.xlsx")
        if f.name.lower().startswith("selected_") and not f.name.startswith("~$")
    )

    if not xlsx_files:
        print(
            f"No Excel files starting with 'selected_' found in {script_dir}. "
            "Add Selected_*.xlsx files and run again."
        )
        return

    frames = []
    for path in xlsx_files:
        df = pd.read_excel(path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Detect sentence column
    sent_col = _sentence_column_name(combined)
    if sent_col not in combined.columns:
        raise KeyError(
            f"No 'sentence' column found. Columns: {list(combined.columns)}"
        )

    # Normalize sentences: remove punctuation, join with |
    combined["sentence"] = combined[sent_col].astype(str).map(_normalize_sentence)

    # If there was a different-named sentence column, drop it to avoid duplicate
    if sent_col != "sentence":
        combined = combined.drop(columns=[sent_col])

    # Word count (number of pipe-separated tokens; 0 for empty)
    combined["words"] = combined["sentence"].apply(
        lambda s: len(s.split("|")) if s and str(s).strip() else 0
    )

    combined.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {len(combined)} rows to {out_path}")


if __name__ == "__main__":
    main()
