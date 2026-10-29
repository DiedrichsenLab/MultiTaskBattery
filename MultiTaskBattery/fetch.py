"""Download task stimuli from Zenodo into a local cache folder.

    mtb fetch                      # stimuli of every task
    mtb fetch reading movie        # only these tasks

Run it once while online (e.g. before going to an offline scanner computer).
Set the MTB_STIMULI environment variable to keep the stimuli somewhere else.
"""
import argparse
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
import pooch

# MultiTaskBattery stimuli v1.0.0 (doi:10.5281/zenodo.23749589). Each version of
# the record has its own ID, so this pins the stimuli to the ones MTB was tested with.
ZENODO_RECORD = '23749589'

# The OS's per-user cache folder (e.g. ~/.cache/MultiTaskBattery on Linux), unless MTB_STIMULI is set
CACHE_DIR = Path(os.environ.get('MTB_STIMULI') or pooch.os_cache('MultiTaskBattery') / 'stimuli')


def record_files():
    """Return {task: md5} for the zips in the Zenodo record (one zip per task)."""
    with urllib.request.urlopen(f'https://zenodo.org/api/records/{ZENODO_RECORD}') as r:
        files = json.load(r)['files']
    return {f['key'][:-len('.zip')]: f['checksum'].split(':')[1] for f in files if f['key'].endswith('.zip')}


def fetch(tasks=None):
    """Download and unpack the stimuli of the given tasks (default: all) into CACHE_DIR.

    A task already unpacked from the same zip is skipped, so fetch can be rerun
    safely; a task unpacked from an older zip is replaced.

    Args:
        tasks (list of str): task names; tasks that use no stimuli are skipped.
    Returns:
        Path: the cache folder.
    """
    available = record_files()
    if tasks is None:
        tasks = list(available)
    elif unknown := [t for t in tasks if t not in available]:
        print(f'No stimuli to fetch for: {", ".join(unknown)} (these tasks need none, or are not in MTB)')
        tasks = [t for t in tasks if t in available]
    for task in tasks:
        md5 = available[task]
        marker = CACHE_DIR / task / '.zenodo_md5'
        if marker.exists() and marker.read_text() == md5:
            continue
        print(f'Fetching {task} stimuli ...')
        zip_path = pooch.retrieve(
            f'https://zenodo.org/records/{ZENODO_RECORD}/files/{task}.zip?download=1',
            known_hash=f'md5:{md5}', fname=f'{task}.zip', path=CACHE_DIR / '_zips')
        shutil.rmtree(CACHE_DIR / task, ignore_errors=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(CACHE_DIR)
        marker.write_text(md5)
        os.remove(zip_path)
    return CACHE_DIR


def main():
    parser = argparse.ArgumentParser(prog='mtb', description='MultiTaskBattery command line tools')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('fetch', help='download task stimuli from Zenodo')
    p.add_argument('tasks', nargs='*', help='tasks to fetch (default: all)')
    args = parser.parse_args()
    print(f'Stimuli are in {fetch(args.tasks or None)}')


if __name__ == '__main__':
    main()
