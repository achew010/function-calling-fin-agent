"""Compare two saved validation reports with a paired conversation bootstrap.

Example:
    python 2_evaluations/compare_validation.py \
        --baseline checkpoints/sft/validation_predictions/step-000000.json \
        --candidate checkpoints/sft/validation_predictions/step-000140.json

Only reads saved predictions; never loads a model or the test split. Intervals are
pointwise 95% intervals, not adjusted for choosing among multiple checkpoints.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "1_training" / "1_sft"))
from metrics import compare_validation_reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = compare_validation_reports(
        json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()),
        n_bootstrap=args.n_bootstrap, seed=args.seed,
    )
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
