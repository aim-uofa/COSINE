#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="${SRC_DIR:-../cosine_original}"
OUT_DIR="${OUT_DIR:-../cosine_archives}"
ARCHIVE_MODE="${ARCHIVE_MODE:-full}"
DATASET_MAX_DEPTH="${DATASET_MAX_DEPTH:-4}"

if [[ ! -d "$SRC_DIR" ]]; then
  echo "Source directory does not exist: $SRC_DIR" >&2
  exit 1
fi

case "$ARCHIVE_MODE" in
  full|logs) ;;
  *)
    echo "ARCHIVE_MODE must be 'full' or 'logs'." >&2
    exit 1
    ;;
esac

mkdir -p "$OUT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
BASENAME="cosine_experiment_${ARCHIVE_MODE}_${STAMP}"
WORK_DIR="$OUT_DIR/${BASENAME}.manifest"
FILELIST="$WORK_DIR/filelist.nul"
SKIPPED_DATASETS="$WORK_DIR/skipped_dataset_regular_files.txt"
SKIPPED_BINARIES="$WORK_DIR/skipped_binary_files.txt"
SYMLINKS="$WORK_DIR/symlinks.txt"
SIZES="$WORK_DIR/top_level_sizes.txt"
GIT_INFO="$WORK_DIR/git_info.txt"

mkdir -p "$WORK_DIR"
: > "$FILELIST"
: > "$SKIPPED_DATASETS"
: > "$SKIPPED_BINARIES"

relpath() {
  local path="$1"
  printf '%s' "${path#"$SRC_DIR"/}"
}

is_under_dataset() {
  local rel="$1"
  case "$rel" in
    datasets|datasets/*|inference_fsod/datasets|inference_fsod/datasets/*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

is_binary_record() {
  local rel="$1"
  case "$rel" in
    *.pth|*.pt|*.bin|*.safetensors|*.ckpt|*.onnx|*.npy|*.npz)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

add_entry() {
  local path="$1"
  local rel
  rel="$(relpath "$path")"

  [[ -n "$rel" ]] || return 0
  [[ "$rel" != "$(basename "$OUT_DIR")"* ]] || return 0

  if is_under_dataset "$rel"; then
    if [[ -L "$path" || -d "$path" ]]; then
      printf '%s\0' "$rel" >> "$FILELIST"
    elif [[ -f "$path" ]]; then
      printf '%s\t%s\n' "$(stat -c '%s' "$path" 2>/dev/null || stat -f '%z' "$path")" "$rel" >> "$SKIPPED_DATASETS"
    fi
    return 0
  fi

  if [[ "$ARCHIVE_MODE" == "logs" && -f "$path" ]] && is_binary_record "$rel"; then
    printf '%s\t%s\n' "$(stat -c '%s' "$path" 2>/dev/null || stat -f '%z' "$path")" "$rel" >> "$SKIPPED_BINARIES"
    return 0
  fi

  printf '%s\0' "$rel" >> "$FILELIST"
}

# Keep normal source, configs, outputs, and logs. Dataset directories are handled
# separately so regular dataset files never enter the archive by accident.
while IFS= read -r -d '' path; do
  add_entry "$path"
done < <(find -P "$SRC_DIR" \
  \( -path "$SRC_DIR/datasets" -o -path "$SRC_DIR/inference_fsod/datasets" \) -type d -prune \
  -o -mindepth 1 -print0)

# Re-add dataset directory skeletons and symlinks only. Depth is capped to avoid
# expensive scans if a dataset directory contains real data rather than links.
for dataset_dir in \
  "$SRC_DIR/datasets" \
  "$SRC_DIR/inference_fsod/datasets"; do
  [[ -d "$dataset_dir" ]] || continue
  while IFS= read -r -d '' path; do
    add_entry "$path"
  done < <(find -P "$dataset_dir" -maxdepth "$DATASET_MAX_DEPTH" \( -type d -o -type l -o -type f \) -print0)
done

sort -z -u "$FILELIST" -o "$FILELIST"
sort -nr "$SKIPPED_DATASETS" -o "$SKIPPED_DATASETS" || true
sort -nr "$SKIPPED_BINARIES" -o "$SKIPPED_BINARIES" || true

find -P "$SRC_DIR" -type l -printf '%p -> %l\n' 2>/dev/null | sort > "$SYMLINKS" || true
du -sh --apparent-size "$SRC_DIR"/* 2>/dev/null | sort -h > "$SIZES" || true
(
  cd "$SRC_DIR"
  git rev-parse --show-toplevel >/dev/null 2>&1 || exit 0
  echo "commit: $(git rev-parse HEAD)"
  echo
  git status --short --branch
) > "$GIT_INFO" || true

if command -v zstd >/dev/null 2>&1; then
  ARCHIVE="$OUT_DIR/${BASENAME}.tar.zst"
  TAR_COMPRESS=(--use-compress-program "zstd -T0 -3")
else
  ARCHIVE="$OUT_DIR/${BASENAME}.tar.gz"
  TAR_COMPRESS=(-z)
fi

tar --no-recursion --null -C "$SRC_DIR" "${TAR_COMPRESS[@]}" -cf "$ARCHIVE" -T "$FILELIST"

sha256sum "$ARCHIVE" > "$ARCHIVE.sha256" 2>/dev/null || shasum -a 256 "$ARCHIVE" > "$ARCHIVE.sha256"

cat <<EOF
Archive created:
  $ARCHIVE

Manifest directory:
  $WORK_DIR

Dataset regular files were not archived. See:
  $SKIPPED_DATASETS

Binary records skipped only when ARCHIVE_MODE=logs. See:
  $SKIPPED_BINARIES
EOF
