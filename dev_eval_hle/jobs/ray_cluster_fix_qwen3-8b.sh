#!/bin/bash
#SBATCH --job-name=ray-2n
#SBATCH -p P07
#SBATCH --nodelist=osk-gpu70,osk-gpu71
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --time=6-00:00:00
#SBATCH --mem=1024GB
#SBATCH --output=%x_%j.log                                  # 出力ログ（%x=ジョブ名, %j=ジョブID）


set -eo pipefail

# ================= 1) Modules =================
source /etc/profile.d/modules.sh
module purge
module load cuda/12.6 miniconda/24.7.1-py312
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
module load cudnn/9.6.0
module load nccl/2.24.3

# ================= 2) Binaries =================
ENV_PREFIX="/home/Competition2025/P07/P07U010/.conda/envs/llmbench_origin"
PY="$ENV_PREFIX/bin/python"
RAY="$ENV_PREFIX/bin/ray"
VLLM="$ENV_PREFIX/bin/vllm"

# ================= 3) Network / NCCL =================
export RAY_IFACE="${RAY_IFACE:-enp25s0np0}"
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME="$RAY_IFACE"
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9
export CUDA_DEVICE_MAX_CONNECTIONS=1
ulimit -v unlimited

# ================= 4) Topology =================
nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' '))
head_node=${nodes_array[0]}
port=37173
dashboard_port=$((port + 1))

# Head IPv4 on fixed NIC
head_node_ip=$(srun -N1 -n1 -w "$head_node" bash -lc \
  "ip -4 -o addr show dev '$RAY_IFACE' | awk '{print \$4}' | cut -d/ -f1")
ip_head=$head_node_ip:$port
export ip_head
echo "[INFO] Head IP on $RAY_IFACE → $ip_head"

# ====== 固定公開先 (vLLM エンドポイントを 192.168.1.70:8010 に固定) ======
SERVE_HOST="192.168.1.70"
SERVE_PORT=8010
echo "[INFO] Reserved serve endpoint → ${SERVE_HOST}:${SERVE_PORT}"

# ヘッドIP検証
if [[ "$head_node_ip" != "$SERVE_HOST" ]]; then
  echo "[ERROR] Head IP ($head_node_ip) != SERVE_HOST ($SERVE_HOST). Fix SERVE_HOST or allocation."
  exit 3
fi
# ポート衝突チェック（ヘッドで）
if srun -N1 -n1 -w "$head_node" bash -lc \
  "ss -H -ltn | awk -v p=\":$SERVE_PORT\" '\$4 ~ p {f=1} END{exit !f}'"; then
  echo "[ERROR] Port $SERVE_PORT is already in use on $SERVE_HOST."
  exit 2
fi

# ================= 5) Ray head =================
# Ray head 起動前
srun -N1 -n1 -w "$head_node" bash -lc "\
  export VLLM_HOST_IP=$head_node_ip; \
  export RAY_TMPDIR=/tmp/r\${SLURM_JOB_ID}; mkdir -p \$RAY_TMPDIR; chmod 700 \$RAY_TMPDIR; export TMPDIR=\$RAY_TMPDIR; \
  $RAY start --head \
    --node-ip-address=$head_node_ip \
    --port=$port \
    --dashboard-port=$dashboard_port --dashboard-host=0.0.0.0 \
    --temp-dir=\$RAY_TMPDIR \
    --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=$SLURM_GPUS_PER_NODE --block" &
sleep 35

# ================= 6) Ray workers =================
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "[INFO] Launching worker on $node_i ..."
  # Ray worker 起動ループ内
  srun -N1 -n1 -w "$node_i" bash -lc "\
    ip=\$(ip -4 -o addr show dev '$RAY_IFACE' | awk '{print \$4}' | cut -d/ -f1); \
    export VLLM_HOST_IP=\$ip; \
    export RAY_TMPDIR=/tmp/r\${SLURM_JOB_ID}; mkdir -p \$RAY_TMPDIR; chmod 700 \$RAY_TMPDIR; export TMPDIR=\$RAY_TMPDIR; \
    $RAY start --address $ip_head --node-ip-address=\$ip \
      --temp-dir=\$RAY_TMPDIR \
      --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=$SLURM_GPUS_PER_NODE --block" &
        
  sleep 5
done
sleep 15

# ================= 7) Connectivity test =================
srun --overlap -N1 -n1 -c1 --gpus=0 -w "$head_node" bash -lc "\
$PY - <<'PY'
import os, json, time, ray
time.sleep(2)
ray.init(address=os.environ['ip_head'])
print(json.dumps({
  'nodes': len(ray.nodes()),
  'detail': [{'host': n['NodeManagerHostname'], 'alive': n['Alive']} for n in ray.nodes()]
}, indent=2))
ray.shutdown()
PY"

# ================= 7.1) vLLM serve を同割当内で起動 =================
TP=$((SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE))   # 例: 16
echo "[INFO] Launching vLLM on ${SERVE_HOST}:${SERVE_PORT} with TP=${TP} ..."
srun --overlap -N1 -n1 -w "$head_node" bash -lc "\
  export RAY_ADDRESS='$ip_head'; \
  export VLLM_HOST_IP='$head_node_ip'; \
  $VLLM serve Qwen/Qwen3-8B \
    --distributed-executor-backend ray \
    --tensor-parallel-size 16 \
    --pipeline-parallel-size 1 \
    --host \"$SERVE_HOST\" --port \"$SERVE_PORT\" \
    --served-model-name qwen3-8b --max-model-len 8192" &


# ================= 8) Watchdog =================
ray_pids=($(jobs -pr))
echo "[INFO] Waiting on Ray/vLLM daemons: ${ray_pids[*]}"

health_ray () { "$RAY" status --address "$ip_head" >/dev/null 2>&1; }
health_vllm () { curl -sf --max-time 5 "http://${SERVE_HOST}:${SERVE_PORT}/health" >/dev/null; }

# vLLM の初回起動は時間がかかる: 最大 45 分待つ（大モデルや初回DLを考慮）
first_ok=0
start_ts=$(date +%s)
warmup_sec=$((45*60))

sleep 20
while true; do
  # 1) バックグラウンドの生存確認
  for pid in "${ray_pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[ERROR] background process $pid exited."
      exit 1
    fi
  done

  # 2) Ray 健康
  if ! health_ray; then
    echo "[ERROR] Ray status unhealthy."
    exit 1
  fi

  # 3) vLLM 健康
  if health_vllm; then
    if [ $first_ok -eq 0 ]; then
      echo "[INFO] vLLM became healthy at http://${SERVE_HOST}:${SERVE_PORT}"
      first_ok=1
    fi
  else
    now=$(date +%s)
    elapsed=$((now - start_ts))
    if [ $first_ok -eq 0 ] && [ $elapsed -lt $warmup_sec ]; then
      echo "[INFO] vLLM warming up (${elapsed}s/${warmup_sec}s) ..."
      sleep 10
      continue
    fi
    echo "[ERROR] vLLM health check failed."
    exit 1
  fi

  sleep 300
done
