#!/bin/bash
# Run the commit-with-proof happy path test
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Starting Hardhat node in background..."
cd "$PROJECT_DIR"
npx hardhat node &
HARDHAT_PID=$!
sleep 3

echo "Running commit-with-proof test..."
npm run test:cp || true

echo "Stopping Hardhat node..."
kill $HARDHAT_PID 2>/dev/null || true
wait $HARDHAT_PID 2>/dev/null || true
echo "Done."
