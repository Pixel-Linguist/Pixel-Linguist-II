#!/usr/bin/env python
"""Rebuild the 393-font rendering library used by PIXEL LINGUIST II.

The fonts themselves are not redistributed in this repository. They are all
freely licensed Google Fonts (SIL OFL 1.1, Apache-2.0, or the Ubuntu Font
Licence); this script fetches them from google-webfonts-helper, which is the
same tool used to assemble the original library, so filenames reproduce exactly.

    python download_font_lib.py --out ./font_lib

Afterwards point the training scripts at that directory via FONTS_DIR.
"""
import argparse
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict

API = "https://gwfh.mranftl.com/api/fonts/{slug}"
UA = {"User-Agent": "Mozilla/5.0"}


def fetch(url, timeout, retries=3):
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(
                urllib.request.Request(url, headers=UA), timeout=timeout
            ).read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"    retry in {wait}s ({exc})")
            time.sleep(wait)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=os.path.join(here, "font_manifest.json"))
    ap.add_argument("--out", default=os.path.join(here, "font_lib"))
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    with open(args.manifest) as fh:
        manifest = json.load(fh)

    # One API call per (category, family): batch all requested variants/subsets.
    groups = defaultdict(lambda: {"subsets": set(), "variants": set(), "files": set()})
    for category, payload in manifest["categories"].items():
        for font in payload["fonts"]:
            key = (category, font["slug"])
            groups[key]["subsets"].update(font["subsets"])
            groups[key]["variants"].add(font["variant"])
            groups[key]["files"].add(font["file"])

    expected = manifest["total_files"]
    print(f"{len(groups)} font families -> {expected} files into {args.out}")

    written, missing, failed = 0, [], []
    for i, ((category, slug), spec) in enumerate(sorted(groups.items()), 1):
        dest = os.path.join(args.out, category)
        os.makedirs(dest, exist_ok=True)

        # Skip families already fully present so the script is resumable.
        if all(os.path.isfile(os.path.join(dest, f)) for f in spec["files"]):
            print(f"[{i}/{len(groups)}] {category}/{slug} already complete")
            written += len(spec["files"])
            continue

        url = API.format(slug=slug) + "?download=zip&formats=ttf" \
            + "&subsets=" + ",".join(sorted(spec["subsets"])) \
            + "&variants=" + ",".join(sorted(spec["variants"]))
        print(f"[{i}/{len(groups)}] {category}/{slug} ({len(spec['files'])} files)")
        try:
            blob = fetch(url, args.timeout)
        except Exception as exc:
            print(f"    FAILED: {exc}")
            failed.append(slug)
            continue

        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            available = set(zf.namelist())
            for name in spec["files"]:
                if name not in available:
                    missing.append(f"{category}/{name}")
                    continue
                with open(os.path.join(dest, name), "wb") as out:
                    out.write(zf.read(name))
                written += 1

    print(f"\n{written}/{expected} files written")
    if missing:
        print(f"{len(missing)} expected file(s) absent upstream (font version may "
              f"have been bumped since the manifest was made):")
        for m in missing[:20]:
            print("   ", m)
    if failed:
        print(f"{len(failed)} family/families failed to download: {failed}")
    if missing or failed:
        print("\nRendering degrades gracefully with a partial library, but scores "
              "will not match the paper exactly.")
        return 1
    print("Font library complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
