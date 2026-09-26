"""Verify this project's six fresh normalization parts, then remove old tails."""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path

EXPECTED = {
    'train_source1': 2_206_821, 'train_source2': 5_034_616,
    'train_source3': 5_285_603, 'test_source1': 1_732_544,
    'test_source2': 4_887_273, 'test_source3': 5_082_316,
}


def inspect(directory: Path):
    stale_paths = []
    for stem, expected in EXPECTED.items():
        paths = sorted(directory.glob(stem + '.part-*.tsv.gz'))
        names = [p.name for p in paths]
        fresh = [directory / f'{stem}.part-{i:02d}.tsv.gz' for i in range(6)]
        stale = [directory / f'{stem}.part-{i:02d}.tsv.gz' for i in range(6, 10)]
        permitted = {p.name for p in fresh + stale}
        if set(names) - permitted or any(not p.is_file() for p in fresh):
            raise ValueError(f'Unexpected or missing normalized parts for {stem}: {names}')
        existing_stale = [p for p in stale if p.exists()]
        if existing_stale and (len(existing_stale) != 4 or
                               min(p.stat().st_mtime_ns for p in fresh) <=
                               max(p.stat().st_mtime_ns for p in existing_stale)):
            raise ValueError(f'Stale parts are not an intact older tail for {stem}')
        count = 0
        for number, part in enumerate(fresh):
            with gzip.open(part, 'rb') as f:
                first = f.readline()
                if number == 0 and not first.startswith(b'entity_id\t'):
                    raise ValueError(f'Missing header in {part}')
                if number > 0 and first.startswith(b'entity_id\t'):
                    raise ValueError(f'Unexpected repeated header in {part}')
                count += int(number > 0)
                count += sum(block.count(b'\n') for block in iter(lambda: f.read(1 << 20), b''))
        if count != expected:
            raise ValueError(f'{stem}: {count:,} rows; expected {expected:,}. No stale parts removed.')
        print(f'PASS {stem}: {count:,} rows; {len(existing_stale)} old tail parts')
        stale_paths.extend(existing_stale)
    return stale_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('normalized_dir')
    parser.add_argument('--remove-stale', action='store_true')
    args = parser.parse_args()
    directory = Path(args.normalized_dir).resolve(strict=True)
    paths = inspect(directory)
    if args.remove_stale:
        for path in paths:
            if path.resolve().parent != directory or not path.name.endswith('.tsv.gz'):
                raise ValueError(f'Unsafe stale path: {path}')
        for path in paths:
            path.unlink()
            print(f'Removed verified stale part: {path.name}')
    else:
        print(f'{len(paths)} verified stale parts would be removed with --remove-stale')


if __name__ == '__main__':
    main()
