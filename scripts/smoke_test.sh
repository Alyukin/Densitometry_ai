#!/usr/bin/env bash
# Smoke-test of a running service: upload -> process -> status -> result -> CSV/XLSX.
# Usage: ./scripts/smoke_test.sh [BASE_URL]   (default http://localhost:8080)
set -euo pipefail

BASE="${1:-http://localhost:${FRONTEND_PORT:-8080}}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SAMPLE="$ROOT/samples/demo_studies.zip"
json() { python3 -c "import sys, json; d = json.load(sys.stdin); print($1)"; }

echo "==> GET $BASE/health"
curl -fsS "$BASE/health" | json 'd["status"], d["processor"], d["processor_version"]'

echo "==> POST /api/v1/studies/upload ($SAMPLE)"
IDS=$(curl -fsS -F "files=@$SAMPLE" "$BASE/api/v1/studies/upload" | json '" ".join(s["id"] for s in d["studies"])')
echo "    studies: $IDS"

echo "==> POST /api/v1/batch/process"
BODY=$(python3 -c "import json,sys; print(json.dumps({'study_ids': sys.argv[1:]}))" $IDS)
curl -fsS -H 'Content-Type: application/json' -d "$BODY" "$BASE/api/v1/batch/process" | json '"accepted:", len(d["accepted"])'

for id in $IDS; do
  for _ in $(seq 1 120); do
    STATUS=$(curl -fsS "$BASE/api/v1/studies/$id/status" | json 'd["status"]')
    [[ "$STATUS" == "completed" || "$STATUS" == "failed" ]] && break
    sleep 0.5
  done
  echo "    $id -> $STATUS"
  curl -fsS "$BASE/api/v1/studies/$id/result" | json '"    rows:", len(d["rows"]), "mock:", d["is_mock"]'
done

FIRST=${IDS%% *}
echo "==> GET /api/v1/studies/$FIRST/download?format=csv"
curl -fsS "$BASE/api/v1/studies/$FIRST/download?format=csv"
echo "==> GET /api/v1/batch/download?format=xlsx"
curl -fsS -o /tmp/densitometry_smoke.xlsx "$BASE/api/v1/batch/download?format=xlsx"
echo "    saved /tmp/densitometry_smoke.xlsx ($(wc -c < /tmp/densitometry_smoke.xlsx) bytes)"
echo "OK"
