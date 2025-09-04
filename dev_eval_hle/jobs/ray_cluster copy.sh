#!/bin/bash
#SBATCH --job-name=ray-2n
#SBATCH -p P07
#SBATCH --nodelist=osk-gpu70,osk-gpu71
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=6-00:00:00
#SBATCH --mem=512GB
#SBATCH --output=%x_%j.log                                  # 出力ログ（%x=ジョブ名, %j=ジョブID）


set -eo pipefail

# 1) Modules
source /etc/profile.d/modules.sh
module purge
module load cuda/12.6 miniconda/24.7.1-py312
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
module load cudnn/9.6.0
module load nccl/2.24.3

# 2) Binaries (absolute paths)
ENV_PREFIX="/home/Competition2025/P07/P07U010/.conda/envs/llmbench_origin"
PY="$ENV_PREFIX/bin/python"
RAY="$ENV_PREFIX/bin/ray"

# 3) Network / NCCL (minimal)
export RAY_IFACE="${RAY_IFACE:-enp25s0np0}"
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME="$RAY_IFACE"
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9
export CUDA_DEVICE_MAX_CONNECTIONS=1
ulimit -v unlimited

# 4) Topology
nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' '))
head_node=${nodes_array[0]}
port=37173
dashboard_port=$((port + 1))

# 固定公開先（vLLMサーバ用）
SERVE_HOST="${SERVE_HOST:-192.168.1.70}"
SERVE_PORT="${SERVE_PORT:-8010}"
echo "[INFO] Reserved serve endpoint → ${SERVE_HOST}:${SERVE_PORT}"

# Head IPv4 on fixed NIC
head_node_ip=$(srun -N1 -n1 -w "$head_node" bash -lc \
  "ip -4 -o addr show dev '$RAY_IFACE' | awk '{print \$4}' | cut -d/ -f1")
ip_head=$head_node_ip:$port
export ip_head
echo "[INFO] Head IP on $RAY_IFACE → $ip_head"

# 4.1) ヘッドIP検証（固定先と一致必須）
if [[ "$head_node_ip" != "$SERVE_HOST" ]]; then
  echo "[ERROR] Head IP ($head_node_ip) != SERVE_HOST ($SERVE_HOST). Fix SERVE_HOST or allocation."
  exit 3
fi

# 4.2) ポート衝突チェック（ヘッドノードで 8010）
if srun -N1 -n1 -w "$head_node" bash -lc \
  "ss -H -ltn | awk -v p=\":$SERVE_PORT\" '\$4 ~ p {exit 0} END{exit 1}'"; then
  echo "[ERROR] Port $SERVE_PORT is already in use on $SERVE_HOST."
  exit 2
fi

# 5) Ray head
srun -N1 -n1 -w "$head_node" bash -lc "\
  export RAY_TMPDIR=/tmp/r\${SLURM_JOB_ID}; mkdir -p \$RAY_TMPDIR; chmod 700 \$RAY_TMPDIR; export TMPDIR=\$RAY_TMPDIR; \
  $RAY start --head \
    --node-ip-address=$head_node_ip \
    --port=$port \
    --dashboard-port=$dashboard_port --dashboard-host=0.0.0.0 \
    --temp-dir=\$RAY_TMPDIR \
    --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=$SLURM_GPUS_PER_NODE --block" &
sleep 35

# 6) Ray workers（可変ノード数に対応）
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "[INFO] Launching worker on $node_i ..."
  srun -N1 -n1 -w "$node_i" bash -lc "\
    export RAY_TMPDIR=/tmp/r\${SLURM_JOB_ID}; mkdir -p \$RAY_TMPDIR; chmod 700 \$RAY_TMPDIR; export TMPDIR=\$RAY_TMPDIR; \
    ip=\$(ip -4 -o addr show dev '$RAY_IFACE' | awk '{print \$4}' | cut -d/ -f1); \
    $RAY start --address $ip_head --node-ip-address=\$ip \
      --temp-dir=\$RAY_TMPDIR \
      --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=$SLURM_GPUS_PER_NODE --block" &
  sleep 5
done
sleep 15

# 7) Connectivity test
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

# 8) Watchdog
ray_pids=($(jobs -pr))
echo "[INFO] Waiting on Ray daemons: ${ray_pids[*]}"

health_check () { "$RAY" status --address "$ip_head" >/dev/null 2>&1; }
sleep 20

while true; do
  for pid in "${ray_pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[ERROR] Ray process $pid exited."
      exit 1
    fi
  done
  if ! health_check; then
    echo "[ERROR] Ray status unhealthy."
    exit 1
  fi
  sleep 300
done
