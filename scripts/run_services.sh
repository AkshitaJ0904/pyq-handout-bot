#!/usr/bin/env bash
# Exercise 4 — starts all four services locally (no Docker) for testing.
# Data Service (5002), Retrieval Service (5003), LLM Service (5004),
# Orchestrator (5001). Logs go to logs/<service>.log. Ctrl+C stops all.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs

PY=venv/bin/python
PIDS=()

cleanup() {
  echo "Stopping services..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

$PY services/data_service.py > logs/data_service.log 2>&1 &
PIDS+=($!)
$PY services/llm_service.py > logs/llm_service.log 2>&1 &
PIDS+=($!)
sleep 2

$PY services/retrieval_service.py > logs/retrieval_service.log 2>&1 &
PIDS+=($!)
sleep 1

$PY services/orchestrator.py > logs/orchestrator.log 2>&1 &
PIDS+=($!)

echo "All services started:"
echo "  data_service       -> http://localhost:5002 (logs/data_service.log)"
echo "  llm_service         -> http://localhost:5004 (logs/llm_service.log)"
echo "  retrieval_service   -> http://localhost:5003 (logs/retrieval_service.log)"
echo "  orchestrator (main) -> http://localhost:5001 (logs/orchestrator.log)"
echo "Press Ctrl+C to stop all."
wait
