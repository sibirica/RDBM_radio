#!/usr/bin/env bash
# Build dataset.json from paired noisy/ground_truth files under DATA_DIR.
# Searches DATA_DIR + one child level only (-maxdepth 2); nested folders
# are ignored — run this script on them separately.
#
#   ./organize_radio.sh [/path/to/data]
#   FORMAT=npy ./organize_radio.sh /path/to/data
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$(cd "${1:-${SCRIPT_DIR}/fix_test_realistic_v2}" && pwd)"
DATASET_JSON="${DATA_DIR}/dataset.json"
FORMAT="${FORMAT:-png}"

if [[ "${FORMAT}" != "png" && "${FORMAT}" != "npy" ]]; then
  echo "Error: FORMAT must be png or npy, got: ${FORMAT}" >&2
  exit 1
fi

EXT="${FORMAT}"
NOISY_SUFFIX="_noisy.${EXT}"
GT_SUFFIX="_ground_truth.${EXT}"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "Error: data directory does not exist: ${DATA_DIR}" >&2
  exit 1
fi

: > "${DATASET_JSON}"
count=0
missing=0

while IFS= read -r noisy; do
  [[ -z "${noisy}" ]] && continue
  base="$(basename "${noisy}")"
  stem="${base%${NOISY_SUFFIX}}"
  gt="$(dirname "${noisy}")/${stem}${GT_SUFFIX}"
  if [[ ! -f "${gt}" ]]; then
    echo "Warning: missing ground truth for ${noisy}" >&2
    missing=$((missing + 1))
    continue
  fi
  printf '{"input": "%s", "target": "%s"}\n' \
    "${noisy#${DATA_DIR}/}" "${gt#${DATA_DIR}/}" >> "${DATASET_JSON}"
  count=$((count + 1))
done < <(find "${DATA_DIR}" -maxdepth 2 -type f -name "*${NOISY_SUFFIX}" | sort -V)

if [[ ${count} -eq 0 ]]; then
  echo "Error: no *${NOISY_SUFFIX} files found under ${DATA_DIR} (maxdepth 2)" >&2
  exit 1
fi
if [[ ${missing} -gt 0 ]]; then
  echo "Error: ${missing} noisy file(s) had no matching *${GT_SUFFIX}" >&2
  exit 1
fi

gt_count="$(find "${DATA_DIR}" -maxdepth 2 -type f -name "*${GT_SUFFIX}" | wc -l)"
orphan_gt=0
while IFS= read -r gt; do
  [[ -z "${gt}" ]] && continue
  base="$(basename "${gt}")"
  stem="${base%${GT_SUFFIX}}"
  noisy="$(dirname "${gt}")/${stem}${NOISY_SUFFIX}"
  if [[ ! -f "${noisy}" ]]; then
    echo "Warning: ground truth has no noisy pair: ${gt}" >&2
    orphan_gt=$((orphan_gt + 1))
  fi
done < <(find "${DATA_DIR}" -maxdepth 2 -type f -name "*${GT_SUFFIX}" | LC_ALL=C sort)

if [[ ${orphan_gt} -gt 0 ]]; then
  echo "Error: ${orphan_gt} ground_truth file(s) without matching noisy" >&2
  exit 1
fi
if [[ "${gt_count}" -ne "${count}" ]]; then
  echo "Error: pair count mismatch (noisy=${count}, ground_truth=${gt_count})" >&2
  exit 1
fi

nested="$(find "${DATA_DIR}" -mindepth 3 -type f -name "*${NOISY_SUFFIX}" 2>/dev/null | wc -l)"
echo "Wrote ${DATASET_JSON} (${count} pairs, maxdepth 2)"
[[ "${nested}" -gt 0 ]] && echo "Note: ignored ${nested} deeper *${NOISY_SUFFIX} file(s)."
echo "OK."
