#!/bin/bash
cd "$(dirname "$0")"
source env.sh
F="grep -v -E ^(W0000|I0000|WARNING)"
echo "=== 300 rest $(date +%T)"
$PY run.py --size 300 --surfaces grf6 barrier floor --max-steps 5000 --repeats 1 2>&1 | $F
echo "=== recompile 100 $(date +%T)"
$PY recompile_test.py 100 2>&1 | $F
echo "=== ejax 300 $(date +%T)"
$PY run_ejax.py 300 5000 uniform grf4 grf6 floor 2>&1 | $F
echo "=== 1000 $(date +%T)"
$PY run.py --size 1000 --surfaces uniform grf4 grf6 floor --variants B B_fine C D E --max-steps 1200 --repeats 1 2>&1 | $F
echo "=== ejax 1000 $(date +%T)"
$PY run_ejax.py 1000 1200 uniform grf6 2>&1 | $F
echo "=== ALL DONE $(date +%T)"
