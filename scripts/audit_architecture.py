"""Audit existing train/development metadata without importing model libraries."""
import argparse
from pathlib import Path
from scripts.fit_reference import TRAIN_SHA, DEV_SHA
from qwen_one_two.training.architecture_audit import audit
from qwen_one_two.training.inputs import canonical
from qwen_one_two.training.release import load_release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    releases = root / 'data/training-collections-20261105/releases'
    report = audit(load_release(releases / 'train', expected_hash=TRAIN_SHA),
                   load_release(releases / 'development', expected_hash=DEV_SHA))
    Path(args.output).write_bytes(canonical(report))
    print(canonical({k: report[k] for k in ('collections', 'receiver_family_overlap', 'receiver_permitted_input_overlap')}).decode())


if __name__ == '__main__':
    main()
