#!/usr/bin/env bash
# Submits a topic, polls until finished, downloads the PDF.
#   BASE=http://127.0.0.1:8100 scripts/smoke-test.sh "topic"    # host development (chapter 09)
#   scripts/smoke-test.sh "topic"                                # containers behind Caddy (chapter 10)
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=${BASE:-https://localhost}
TOKEN=$(head -n1 secrets/api_tokens | cut -d: -f2)
TOPIC=${1:-}
[ -n "$TOPIC" ] || TOPIC="How does the Linux CFS scheduler work"

CURL=(curl -sS --fail-with-body -H "Authorization: Bearer $TOKEN")
if [ -f infra/caddy-root.crt ]; then CURL+=(--cacert infra/caddy-root.crt); else CURL+=(-k); fi

JOB=$("${CURL[@]}" -H "Content-Type: application/json" \
  -d "$(jq -n --arg t "$TOPIC" '{topic: $t}')" "$BASE/research" | jq -r .job_id)
echo "job: $JOB"

while :; do
  STATUS=$("${CURL[@]}" "$BASE/jobs/$JOB" | jq -r .status)
  echo "$(date +%T) status: $STATUS"
  case "$STATUS" in
    done|done_low_confidence) break ;;
    failed|rejected) "${CURL[@]}" "$BASE/jobs/$JOB" | jq .; exit 1 ;;
  esac
  sleep 5
done

"${CURL[@]}" "$BASE/jobs/$JOB" | jq .
"${CURL[@]}" -o "report-$JOB.pdf" "$BASE/jobs/$JOB/pdf"
echo "saved report-$JOB.pdf"