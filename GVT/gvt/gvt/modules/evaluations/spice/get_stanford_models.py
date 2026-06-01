#!/usr/bin/python
# This script downloads the Stanford CoreNLP models.
import os
import shutil
from urllib.request import urlretrieve
from zipfile import ZipFile

CORENLP = 'stanford-corenlp-full-2015-12-09'
SPICELIB = 'lib'
JAR = 'stanford-corenlp-3.6.0'
SPICEDIR = os.path.dirname(__file__)
MAVEN_BASE = 'https://repo1.maven.org/maven2/edu/stanford/nlp/stanford-corenlp/3.6.0'


def print_progress(transferred_blocks, block_size, total_size):
    current_mb = transferred_blocks * block_size / 1024 / 1024
    total_mb = total_size / 1024 / 1024
    percent = current_mb / total_mb
    progress_str = "Progress: {:5.1f}M / {:5.1f}M ({:6.1%})"
    print(progress_str.format(current_mb, total_mb, percent), end='\r')


def get_stanford_models():
    lib_dir = os.path.join(SPICEDIR, SPICELIB)
    os.makedirs(lib_dir, exist_ok=True)

    required = ['{}.jar'.format(JAR), '{}-models.jar'.format(JAR)]
    target_paths = [os.path.join(lib_dir, name) for name in required]
    if all(os.path.exists(path) for path in target_paths):
        return

    # Recover from a previous interrupted extraction before downloading again.
    extracted_dir = os.path.join(SPICEDIR, CORENLP)
    for name, target_path in zip(required, target_paths):
        source_path = os.path.join(extracted_dir, name)
        if not os.path.exists(target_path) and os.path.exists(source_path):
            shutil.move(source_path, target_path)

    if all(os.path.exists(path) for path in target_paths):
        shutil.rmtree(extracted_dir, ignore_errors=True)
        return

    # Prefer direct Maven downloads: this avoids downloading and extracting the
    # full CoreNLP zip when only one jar is missing.
    direct_failed = False
    for name, target_path in zip(required, target_paths):
        if os.path.exists(target_path):
            continue
        try:
            print('Downloading {} for SPICE ...'.format(name))
            urlretrieve('{}/{}'.format(MAVEN_BASE, name), target_path, reporthook=print_progress)
            print()
        except Exception:
            direct_failed = True
            if os.path.exists(target_path):
                os.remove(target_path)
            break

    if all(os.path.exists(path) for path in target_paths):
        print('Done.')
        return
    if not direct_failed:
        return

    print('Downloading {} for SPICE ...'.format(JAR))
    url = 'http://nlp.stanford.edu/software/{}.zip'.format(CORENLP)
    zip_file, headers = urlretrieve(url, reporthook=print_progress)
    print()
    print('Extracting {} ...'.format(JAR))

    try:
        with ZipFile(zip_file) as archive:
            names = archive.namelist()
            for name, target_path in zip(required, target_paths):
                if os.path.exists(target_path):
                    continue
                suffix = '/{}'.format(name)
                matches = [member for member in names if member.endswith(suffix)]
                if not matches:
                    raise FileNotFoundError(
                        "Could not find {} inside {}".format(name, zip_file)
                    )
                with archive.open(matches[0]) as source, open(target_path, 'wb') as target:
                    shutil.copyfileobj(source, target)
    finally:
        os.remove(zip_file)
        shutil.rmtree(extracted_dir, ignore_errors=True)

    print('Done.')


if __name__ == '__main__':
    # If run as a script, excute inside the spice/ folder.
    get_stanford_models()
