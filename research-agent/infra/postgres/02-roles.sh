#!/bin/bash
# Least-privilege roles. Runs once, after 01-schema.sql, when the data volume is first created.
set -euo pipefail
API_PW=$(cat /run/secrets/pg_api_password)
WORKER_PW=$(cat /run/secrets/pg_worker_password)

psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" <<SQL
CREATE ROLE app_api LOGIN PASSWORD '${API_PW}';
CREATE ROLE app_worker LOGIN PASSWORD '${WORKER_PW}';

REVOKE ALL ON DATABASE ${POSTGRES_DB} FROM PUBLIC;
GRANT CONNECT ON DATABASE ${POSTGRES_DB} TO app_api, app_worker;

GRANT SELECT, INSERT ON jobs TO app_api;
GRANT SELECT ON eval_results TO app_api;

GRANT SELECT, UPDATE ON jobs TO app_worker;
GRANT SELECT, INSERT, DELETE ON chunks TO app_worker;
GRANT SELECT, INSERT, UPDATE ON eval_results TO app_worker;
GRANT USAGE ON SEQUENCE chunks_id_seq TO app_worker;

ALTER SCHEMA lg OWNER TO app_worker;
ALTER ROLE app_worker SET search_path = lg, public;
SQL