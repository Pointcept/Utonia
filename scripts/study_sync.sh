#!/usr/bin/env bash
# Download or upload the git-ignored study/ folder from or to S3.
#
#   scripts/study_sync.sh pull             # S3 -> study/
#   scripts/study_sync.sh push --dryrun    # preview study/ -> S3
#   scripts/study_sync.sh push             # study/ -> S3
#
# Extra arguments go to `aws s3 sync` (e.g. --dryrun, --delete, --profile NAME).
# Needs the AWS CLI with access to the bucket (for SSO: aws sso login).
set -euo pipefail

S3_URI="s3://thindev/pointcept-utonia-demo/study"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/study"
EXCLUDES=(--exclude "*__pycache__/*")

usage() {
    sed -n '2,9s/^# \{0,1\}//p' "${BASH_SOURCE[0]}"
    exit 1
}

action="${1:-}"
[[ $# -gt 0 ]] && shift
case "$action" in
    pull)
        mkdir -p "$LOCAL_DIR"
        aws s3 sync "$S3_URI" "$LOCAL_DIR" "${EXCLUDES[@]}" "$@"
        ;;
    push)
        [[ -d "$LOCAL_DIR" ]] || { echo "No study folder at $LOCAL_DIR" >&2; exit 1; }
        aws s3 sync "$LOCAL_DIR" "$S3_URI" "${EXCLUDES[@]}" "$@"
        ;;
    *)
        usage
        ;;
esac
