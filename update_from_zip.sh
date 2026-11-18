#!/usr/bin/env bash
# Update this project from a downloaded zip. Your data/ and venv/ are kept; only code, website
# and settings files are replaced. New correction rows in data/curation/overrides.csv are added.
#
# Usage, inside the project folder (this runs the NEW script straight from the zip):
#   unzip -p "$HOME/Downloads/icwsm-datahub.zip" icwsm-datahub/update_from_zip.sh | bash -s -- "$HOME/Downloads/icwsm-datahub.zip"
set -euo pipefail
ZIP="${1:?Give the path to the zip, in quotes}"
[ -f "$ZIP" ] || { echo "Zip not found: $ZIP"; exit 1; }
[ -d data ] && [ -d scripts ] || { echo "Run this inside your project folder (the one with data/ and scripts/)"; exit 1; }
STAMP=$(date +%Y%m%d-%H%M)
BK="../icwsm-code-backup-$STAMP"
echo "1/5 Backing up code and settings to $BK (data/ is not touched)"
mkdir -p "$BK" && cp -R icwsm_datahub scripts tests docs "$BK"/ 2>/dev/null || true
cp data/curation/overrides.csv "$BK"/ 2>/dev/null || true
echo "2/5 Unzipping"
TMP=$(mktemp -d) && unzip -q "$ZIP" -d "$TMP"
SRC="$TMP/icwsm-datahub"
echo "3/5 Copying code, website and settings"
for d in icwsm_datahub scripts tests; do rsync -a --delete "$SRC/$d/" "./$d/"; done
mkdir -p docs .github && rsync -a "$SRC/docs/" docs/ --exclude data/ && rsync -a "$SRC/.github/" .github/
for f in requirements.txt Makefile README.md update_from_zip.sh .gitignore LICENSE LICENSE-DATA.md; do
  [ -f "$SRC/$f" ] && cp "$SRC/$f" .
done
[ -f release.json ] || cp "$SRC/release.json" .          # never overwrite your filled-in release.json
if [ -f "$SRC/data/curation/overrides.csv" ]; then        # add correction rows you don't have yet
  mkdir -p data/curation; touch data/curation/overrides.csv
  [ -s data/curation/overrides.csv ] || head -1 "$SRC/data/curation/overrides.csv" > data/curation/overrides.csv
  [ -n "$(tail -c1 data/curation/overrides.csv)" ] && echo >> data/curation/overrides.csv
  added=0
  while IFS= read -r line || [ -n "$line" ]; do
    [ -n "$line" ] || continue
    grep -qxF -- "$line" data/curation/overrides.csv || { echo "$line" >> data/curation/overrides.csv; added=$((added+1)); }
  done < <(tail -n +2 "$SRC/data/curation/overrides.csv")
  echo "    $added new correction row(s) added to data/curation/overrides.csv"
fi
echo "4/5 Installing packages and running tests"
pip install -q -r requirements.txt
python -m pytest -q tests 2>/dev/null | tail -1
echo "5/5 Rebuilding the catalog, website data and share image"
python scripts/05_build_catalog.py | tail -3
python scripts/08_social_card.py
echo "Done. Extracted papers kept: $(ls data/extracted 2>/dev/null | wc -l | tr -d ' ')"
