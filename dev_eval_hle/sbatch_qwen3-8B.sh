#!/bin/bash
#SBATCH --job-name=vllm-serve
#SBATCH -p P07
#SBATCH --nodelist=osk-gpu70
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1          # ← 0だとvLLM 0.9.2がデバイス推定で落ちる
#SBATCH --cpus-per-task=2
#SBATCH --time=6-00:00:00
#SBATCH --mem=4G
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err

set -eo pipefail

# Binaries
ENV_PREFIX="/home/Competition2025/P07/P07U010/.conda/envs/llmbench_origin"
VLLM="$ENV_PREFIX/bin/vllm"
PY="$ENV_PREFIX/bin/python"

# Ray
RAY_ADDRESS="${RAY_ADDRESS:-192.168.1.70:37173}"

# 固定公開アドレス
SERVE_HOST="${SERVE_HOST:-192.168.1.70}"
SERVE_PORT="${SERVE_PORT:-8010}"
BASE="http://${SERVE_HOST}:${SERVE_PORT}"

# ポート衝突チェック（ローカルでOK）
if ss -H -ltn | awk -v p=":${SERVE_PORT}" '$4 ~ p {found=1} END{exit !found}'; then
  echo "[ERROR] Port $SERVE_PORT is already in use on $SERVE_HOST."
  exit 2
fi

# vLLM 起動（バックグラウンド）
"$VLLM" serve Qwen/Qwen3-8B \
  --distributed-executor-backend ray \
  --ray-address "$RAY_ADDRESS" \
  --tensor-parallel-size 16 \
  --pipeline-parallel-size 1 \
  --host "$SERVE_HOST" \
  --port "$SERVE_PORT" \
  --served-model-name qwen3-8b \
  --max-model-len 8192 \
  --device cuda &
SERVE_PID=$!
trap 'kill $SERVE_PID 2>/dev/null || true' EXIT

# ヘルス待ち
echo "[INFO] Waiting for vLLM health at $BASE/health ..."
for i in {1..90}; do
  if curl -sf "$BASE/health" >/dev/null; then
    echo "[INFO] vLLM is healthy."
    break
  fi
  sleep 2
done || { echo "[ERROR] vLLM did not become healthy."; exit 1; }

# テスト（Chat Completions）
echo "[INFO] Running test request..."
curl -sS "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b",
    "messages": [{"role":"user","content":"テスト: 1行で自己紹介して。"}],
    "max_tokens": 32,
    "temperature": 0.0
  }' | "$PY" - <<'PY'
import sys, json
try:
    resp=json.load(sys.stdin)
    print("\n[TEST RESULT]\n", resp["choices"][0]["message"]["content"])
except Exception:
    sys.stdout.write(sys.stdin.read())
PY

# 常駐
wait $SERVE_PID
