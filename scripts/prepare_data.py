#!/usr/bin/env python3
"""Download the official COCO 2017 annotations and extract training captions."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile

URL = 'https://images.cocodataset.org/annotations/annotations_trainval2017.zip'

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='data/captions_train2017.json')
    a = p.parse_args()
    path = Path(a.out)
    if path.exists():
        data = json.loads(path.read_text())
        if not data.get('annotations') or 'caption' not in data['annotations'][0]:
            raise ValueError(f'Not a COCO caption annotation file: {path}')
        print(f'Already available: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {URL}', flush=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as temp:
        archive = Path(temp)/'annotations.zip'
        with urlopen(URL, timeout=120) as response, archive.open('wb') as output:
            shutil.copyfileobj(response, output)
        with ZipFile(archive) as z:
            with z.open('annotations/captions_train2017.json') as source:
                extracted = Path(temp)/'captions.json'
                with extracted.open('wb') as output:
                    shutil.copyfileobj(source, output)
        data = json.loads(extracted.read_text())
        if not data.get('annotations'):
            raise ValueError('Downloaded caption file is empty')
        extracted.replace(path)
    print(f'Prepared {len(data["annotations"])} captions: {path}')

if __name__ == '__main__':
    main()
