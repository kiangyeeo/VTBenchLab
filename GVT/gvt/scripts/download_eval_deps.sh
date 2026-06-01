#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"

tokenizer_dir="${project_root}/gvt/modules/evaluations/tokenizer"
spice_dir="${project_root}/gvt/modules/evaluations/spice"
spice_lib_dir="${spice_dir}/lib"

mkdir -p "${tokenizer_dir}" "${spice_dir}" "${spice_lib_dir}"

download_file() {
  local url="$1"
  local output="$2"
  if [[ -f "${output}" ]]; then
    echo "[skip] exists: ${output}"
    return
  fi
  echo "[download] ${url}"
  python -c "from urllib.request import urlretrieve; urlretrieve('${url}', '${output}')"
  echo "[ok] ${output}"
}

download_file \
  "https://github.com/tylin/coco-caption/raw/master/pycocoevalcap/tokenizer/stanford-corenlp-3.4.1.jar" \
  "${tokenizer_dir}/stanford-corenlp-3.4.1.jar"

download_file \
  "https://github.com/tylin/coco-caption/raw/master/pycocoevalcap/spice/spice-1.0.jar" \
  "${spice_dir}/spice-1.0.jar"

python - <<PY
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

spice_dir = Path("${spice_dir}")
lib_dir = Path("${spice_lib_dir}")
bundle = spice_dir / "SPICE-1.0.zip"
required = [
    "ejml-0.23.jar",
    "slf4j-api-1.7.12.jar",
    "slf4j-simple-1.7.21.jar",
    "lmdbjni-0.4.6.jar",
    "lmdbjni-linux64-0.4.6.jar",
    "lmdbjni-osx64-0.4.6.jar",
    "lmdbjni-win64-0.4.6.jar",
    "fst-2.47.jar",
    "jackson-core-2.5.3.jar",
    "javassist-3.19.0-GA.jar",
    "objenesis-2.4.jar",
    "guava-19.0.jar",
    "json-simple-1.1.1.jar",
    "Meteor-1.5.jar",
    "SceneGraphParser-1.0.jar",
]

missing = [name for name in required if not (lib_dir / name).is_file()]
if missing:
    still_missing = []
    github_base = "https://github.com/tylin/coco-caption/raw/master/pycocoevalcap/spice/lib"
    for name in missing:
        target = lib_dir / name
        url = f"{github_base}/{name}"
        try:
            print(f"[download] {url}")
            urlretrieve(url, target)
            print(f"[ok] {target}")
        except Exception as exc:
            if target.exists():
                target.unlink()
            print(f"[warn] direct download failed for {name}: {exc}")
            still_missing.append(name)

    if still_missing:
        if not bundle.is_file():
            url = "https://panderson.me/images/SPICE-1.0.zip"
            print(f"[download] {url}")
            urlretrieve(url, bundle)
            print(f"[ok] {bundle}")

        with ZipFile(bundle) as archive:
            members = archive.namelist()
            for name in still_missing:
                matches = [member for member in members if member.endswith("/lib/" + name)]
                if not matches:
                    raise FileNotFoundError(f"Could not find lib/{name} in {bundle}")
                target = lib_dir / name
                print(f"[extract] {name}")
                with archive.open(matches[0]) as src, target.open("wb") as dst:
                    dst.write(src.read())
else:
    print("[skip] SPICE third-party lib jars already exist")
PY

echo "[download] Stanford CoreNLP models for SPICE, if missing"
python "${spice_dir}/get_stanford_models.py"

echo "[done] COCO Caption Java evaluation dependencies are ready."
