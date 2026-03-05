#!/usr/bin/env bash
# Training monitor - shows progress, memory, GPU stats
set -euo pipefail

LOGFILE=${1:-/tmp/claude-1001/-home-ianblenke-ultravox-0-7-qwen3-5-9B/tasks/b0hilt6t8.output}

echo "=== $(date) ==="
echo "=== RAM ==="
free -h | grep -E "Mem|Swap"
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
echo "=== Training ==="
grep -E "loss" "$LOGFILE" | tail -1
STEP=$(grep -oP '\d+(?=/781352)' "$LOGFILE" | tail -1)
SPEED=$(grep -oP '\d+\.\d+s/it' "$LOGFILE" | tail -1)
echo "Step: $STEP / 781,352 @ $SPEED"
REMAINING=$(python3 -c "s=781352-${STEP:-0}; spd=float('${SPEED:-2.0}'.replace('s/it','')); t=s*spd/3600; print(f'{t:.0f} hours ({t/24:.1f} days)')")
echo "ETA: $REMAINING"
