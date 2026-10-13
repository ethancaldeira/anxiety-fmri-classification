#!/usr/bin/env python
"""
make_gsp_inputs.py — build the GSP subject list and train/val/test split.

Writes the two GSP input files that the pipeline reads at runtime:

    data/subjects.txt     CSV: participant_id, STAI_tAnxiety
    data/gsp_split.json   {"train": [...], "val": [...], "test": [...]}

Neither file is distributed with the repository; both are derived from the
GSP phenotype table (the STAI-T scores are part of the GSP restricted
phenotypic data, available under the GSP restricted data use terms).

Recipe
------
1. Read the participants TSV (columns `participant_id`, `STAI_tAnxiety`).
2. Keep the subjects with a preprocessed BOLD. Either pass the list of
   subjects whose C-PAC output contains the
   `space-MNI152NLin6ASym_reg-1_desc-preproc_bold` file (--preprocessed),
   or drop the subjects named in --exclude. By default --exclude names the
   two GSP subjects for which C-PAC functional preprocessing did not
   complete (it stopped at nuisance regression, so no preprocessed BOLD
   was written): Sub0105_S1 and Sub1382_S1. This gives 877 subjects.
3. Drop subjects with a missing STAI_tAnxiety value and sort by
   participant_id.
4. Label = STAI_tAnxiety >= --cutoff (36, the cohort median).
5. Stratified on that label with scikit-learn's train_test_split and
   random_state = --seed (42): hold out --test-size (0.20) of the subjects
   as the test set, then --val-size (0.15) of the remainder as the
   validation set. This gives 595 / 106 / 176 subjects.

Usage
-----
    python scripts/make_gsp_inputs.py --participants /path/to/GSP_participants.tsv

    # or with an explicit list of subjects that have a preprocessed BOLD
    # (one ID per line, SubNNNN_S1 or sub-NNNN)
    python scripts/make_gsp_inputs.py --participants /path/to/GSP_participants.tsv \\
        --preprocessed preprocessed_subjects.txt

    # compare with existing files instead of writing
    python scripts/make_gsp_inputs.py --participants /path/to/GSP_participants.tsv --check
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / 'data'

# Subjects for which C-PAC functional preprocessing did not complete.
DEFAULT_EXCLUDE = ['Sub0105_S1', 'Sub1382_S1']


def _to_participant_id(raw: str) -> str:
    """Accept 'SubNNNN_S1' or the BIDS form 'sub-NNNN'; return 'SubNNNN_S1'."""
    raw = raw.strip()
    m = re.fullmatch(r'sub-(\d+)', raw)
    return f'Sub{int(m.group(1)):04d}_S1' if m else raw


def build_subjects(participants: Path, preprocessed: Path | None,
                   exclude: list[str]) -> pd.DataFrame:
    df = pd.read_csv(participants, sep='\t')
    missing = {'participant_id', 'STAI_tAnxiety'} - set(df.columns)
    if missing:
        sys.exit(f"ERROR: {participants} lacks column(s) {sorted(missing)}")
    df = df[['participant_id', 'STAI_tAnxiety']].copy()

    if preprocessed is not None:
        with open(preprocessed) as f:
            keep = {_to_participant_id(line) for line in f if line.strip()}
        unknown = keep - set(df.participant_id)
        if unknown:
            print(f"  [warn] {len(unknown)} ID(s) in {preprocessed} are not in "
                  f"the participants table and are ignored")
        df = df[df.participant_id.isin(keep)]
    else:
        df = df[~df.participant_id.isin(set(exclude))]

    n_before = len(df)
    df = df.dropna(subset=['STAI_tAnxiety'])
    if len(df) < n_before:
        print(f"  dropped {n_before - len(df)} subject(s) without STAI_tAnxiety")
    return df.sort_values('participant_id').reset_index(drop=True)


def build_split(subjects: pd.DataFrame, cutoff: float, seed: int,
                test_size: float, val_size: float) -> dict[str, list[str]]:
    ids = subjects['participant_id'].values
    y = (subjects['STAI_tAnxiety'] >= cutoff).astype(int).values
    ids_trval, ids_test, y_trval, _ = train_test_split(
        ids, y, test_size=test_size, stratify=y, random_state=seed)
    ids_train, ids_val, _, _ = train_test_split(
        ids_trval, y_trval, test_size=val_size, stratify=y_trval, random_state=seed)
    return {
        'train': [str(s) for s in ids_train],
        'val':   [str(s) for s in ids_val],
        'test':  [str(s) for s in ids_test],
    }


def _check(subjects: pd.DataFrame, split: dict, out_dir: Path) -> int:
    rc = 0
    subj_path, split_path = out_dir / 'subjects.txt', out_dir / 'gsp_split.json'
    if subj_path.is_file():
        old = pd.read_csv(subj_path)
        same = (list(old.participant_id) == list(subjects.participant_id)
                and (old.STAI_tAnxiety.values == subjects.STAI_tAnxiety.values).all())
        print(f"  {subj_path}: {'identical' if same else 'DIFFERS'}")
        rc |= not same
    else:
        print(f"  {subj_path}: not found")
    if split_path.is_file():
        with open(split_path) as f:
            old = json.load(f)
        for part in ('train', 'val', 'test'):
            same = set(old.get(part, [])) == set(split[part])
            print(f"  {split_path} [{part}]: {'same subjects' if same else 'DIFFERS'}")
            rc |= not same
    else:
        print(f"  {split_path}: not found")
    return int(rc)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--participants', required=True, type=Path,
                    help='GSP participants TSV with participant_id and STAI_tAnxiety.')
    ap.add_argument('--preprocessed', type=Path, default=None,
                    help='Optional text file listing the subjects with a preprocessed '
                         'BOLD (one ID per line). Overrides --exclude.')
    ap.add_argument('--exclude', nargs='*', default=DEFAULT_EXCLUDE,
                    help=f'Subjects without a preprocessed BOLD (default: {DEFAULT_EXCLUDE}).')
    ap.add_argument('--cutoff', type=float, default=36.0,
                    help='STAI-T cut-off used to stratify the split (default 36).')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--test-size', type=float, default=0.20)
    ap.add_argument('--val-size', type=float, default=0.15,
                    help='Fraction of the non-test subjects used for validation.')
    ap.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR,
                    help=f'Output directory (default: {DEFAULT_OUT_DIR}).')
    ap.add_argument('--check', action='store_true',
                    help='Compare with the files in --out-dir instead of writing.')
    args = ap.parse_args()

    subjects = build_subjects(args.participants, args.preprocessed, args.exclude)
    split = build_split(subjects, args.cutoff, args.seed, args.test_size, args.val_size)
    n_pos = int((subjects.STAI_tAnxiety >= args.cutoff).sum())
    print(f"  subjects: {len(subjects)}  (STAI-T >= {args.cutoff:g}: {n_pos})")
    print(f"  split:    train {len(split['train'])} / val {len(split['val'])} "
          f"/ test {len(split['test'])}")

    if args.check:
        sys.exit(_check(subjects, split, args.out_dir))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    subjects.to_csv(args.out_dir / 'subjects.txt', index=False)
    with open(args.out_dir / 'gsp_split.json', 'w') as f:
        json.dump(split, f, indent=2)
    print(f"  wrote {args.out_dir / 'subjects.txt'}")
    print(f"  wrote {args.out_dir / 'gsp_split.json'}")


if __name__ == '__main__':
    main()
