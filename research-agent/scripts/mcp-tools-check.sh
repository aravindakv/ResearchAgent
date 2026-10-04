#!/usr/bin/env bash
# Chapter 05: exercise every MCP tool through the MCP Inspector CLI (latest), with PASS/FAIL per check.
#
#   scripts/mcp-tools-check.sh          (or: make mcp-check)
#
# Needs: the MCP server running (`make mcp` in another terminal), Node.js (npx), jq, curl.
# Works from any directory: it switches to the repo root itself.
# Overrides: MCP_CHECK_URL (default http://127.0.0.1:8200/mcp), INSPECTOR (the command to run).
set -uo pipefail
cd "$(dirname "$0")/.."

URL=${MCP_CHECK_URL:-http://127.0.0.1:8200/mcp}
BASE=${URL%/mcp}
INSPECTOR=${INSPECTOR:-npx -y @modelcontextprotocol/inspector@latest}
PDF=reports/mcp-check.pdf
PASS=0
FAIL=0

ok()   { echo "  PASS  $1"; PASS=$((PASS + 1)); }
bad()  { echo "  FAIL  $1"; FAIL=$((FAIL + 1)); }
info() { echo "        $1"; }

# ---------------------------------------------------------------- prerequisites
echo "== Prerequisites"
missing=0
for cmd in npx jq curl; do
  command -v "$cmd" >/dev/null || { echo "  missing command: $cmd"; missing=1; }
done
[ -s secrets/mcp_token ] || { echo "  missing secrets/mcp_token: run 'make secrets'"; missing=1; }
if ! curl -sf "$BASE/healthz" >/dev/null; then
  echo "  MCP server not reachable at $BASE: start it with 'make mcp' in another terminal"
  missing=1
fi
[ "$missing" -eq 0 ] || exit 2
echo "  ok: npx, jq, curl, token, server at $BASE"
TOKEN=$(cat secrets/mcp_token)

# call_tool <tool> "<name>=<value>": prints the tool result JSON (stdout only)
call_tool() {
  $INSPECTOR --cli "$URL" --header "Authorization: Bearer $TOKEN" \
    --method tools/call --tool-name "$1" --tool-arg "$2" 2>/dev/null
}
text_of()  { jq -r '.content[0].text // empty' 2>/dev/null; }
error_of() { jq -r '.isError // false' 2>/dev/null; }

# ---------------------------------------------------------------- 0. auth
echo "== 0. Requests without the token are refused"
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$URL")
[ "$code" = "401" ] && ok "POST /mcp without token -> 401" || bad "expected 401 without token, got $code"

# ---------------------------------------------------------------- 1. web_search
echo "== 1. web_search returns JSON results"
out=$(call_tool web_search "query=raft leader election")
count=$(printf '%s' "$out" | text_of | jq 'length' 2>/dev/null || echo 0)
if [ "$(printf '%s' "$out" | error_of)" = "false" ] && [ "${count:-0}" -gt 0 ] 2>/dev/null; then
  ok "$count results"
  printf '%s' "$out" | text_of | jq -r '.[:3][] | "- \(.title) (\(.url))"' | while read -r line; do info "$line"; done
else
  bad "no results: $(printf '%s' "$out" | text_of | head -c 160)"
  info "check secrets/tavily_api_key and the server log in the other terminal"
fi

# ---------------------------------------------------------------- 2. fetch_page (public)
echo "== 2. fetch_page reads a public page"
out=$(call_tool fetch_page "url=https://example.com")
txt=$(printf '%s' "$out" | text_of)
if [ "$(printf '%s' "$out" | error_of)" = "false" ] && [ -n "$txt" ]; then
  ok "got $(printf '%s' "$txt" | wc -c) characters of text"
  info "$(printf '%s' "$txt" | head -c 100 | tr '\n' ' ')..."
else
  bad "fetch failed: ${txt:-no output}"
  info "if checks 3 and 4 pass, this machine probably can't reach the internet from the terminal (proxy/firewall)"
fi

# ---------------------------------------------------------------- 3 and 4. SSRF guard
blocked_check() {  # <title> <url>
  echo "== $1"
  local out txt
  out=$(call_tool fetch_page "url=$2")
  txt=$(printf '%s' "$out" | text_of)
  if [ "$(printf '%s' "$out" | error_of)" = "true" ] && [[ "$txt" == *"non-public address blocked"* ]]; then
    ok "$2 -> blocked"
  else
    bad "$2 was NOT blocked as expected: ${txt:-no output}"
  fi
}
blocked_check "3. fetch_page refuses the cloud metadata address" "http://169.254.169.254/latest/meta-data/"
blocked_check "4. fetch_page refuses loopback (the server itself)" "http://127.0.0.1:8200/healthz"

# ---------------------------------------------------------------- 5. render_pdf
echo "== 5. render_pdf returns a base64 PDF"
mkdir -p reports
call_tool render_pdf "markdown=# Hello" | text_of | base64 -d > "$PDF" 2>/dev/null
if [ "$(head -c 5 "$PDF" 2>/dev/null)" = "%PDF-" ]; then
  ok "saved $PDF ($(wc -c < "$PDF") bytes); open it with: xdg-open $PDF"
else
  bad "no valid PDF returned (check the server log for WeasyPrint errors)"
fi

# ---------------------------------------------------------------- summary
echo
echo "== $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
