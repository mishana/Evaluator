# NeMo Skills Golden Data

This directory contains golden (reference) test data for NeMo Skills integration tests.

## Files

### `gsm8k_math_input.jsonl`
Sample GSM8K math reasoning problems in JSONL format. Each line contains:
- `problem`: The math word problem text
- `expected_answer`: The correct numeric answer
- `answer`: Alias for `expected_answer`

### `multichoice_input.jsonl`
Sample multiple choice questions in JSONL format. Each line contains:
- `question`: The question text
- `choices`: List of answer choices
- `expected_answer`: The correct choice letter (A, B, C, D)
- `answer`: Alias for `expected_answer`

## Usage

These files are used by unit tests in `tests/unit_tests/nemo_skills/` to validate
the benchmark data loading, prompt construction, and scoring functions.
