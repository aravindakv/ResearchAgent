#!/usr/bin/env bash
# Verifies network boundaries and the SSRF guard. Run after `make up`.
set -uo pipefail
cd "$(dirname "$0")/.."

can_reach() {
  docker compose exec -T "$1" python -c \
    "import socket; socket.create_connection(('$2', $3), timeout=3)" >/dev/null 2>&1
}
expect_blocked() { if can_reach "$1" "$2" "$3"; then echo "FAIL  $1 -> $2:$3 is reachable"; else echo "ok    $1 -> $2:$3 blocked"; fi; }
expect_open()    { if can_reach "$1" "$2" "$3"; then echo "ok    $1 -> $2:$3 reachable"; else echo "FAIL  $1 -> $2:$3 unreachable"; fi; }

echo "== Should be blocked"
expect_blocked api    example.com 443
expect_blocked api    mcp         8000
expect_blocked mcp    postgres    5432
expect_blocked mcp    redis       6379
expect_blocked mcp    api         8000
echo "== Should be reachable"
expect_open    worker postgres    5432
expect_open    worker mcp         8000
expect_open    mcp    example.com 443

echo "== SSRF guard"
for url in http://169.254.169.254/ http://127.0.0.1:8000/ http://postgres:5432/ file:///etc/passwd; do
  if docker compose exec -T mcp python -c "from server import assert_public_url; assert_public_url('$url')" >/dev/null 2>&1; then
    echo "FAIL  $url allowed"; else echo "ok    $url rejected"; fi
done