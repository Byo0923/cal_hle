#!/bin/bash
#SBATCH --job-name=ray-2n
#SBATCH -p P07
#SBATCH --nodelist=osk-gpu70,osk-gpu71
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=6-00:00:00
#SBATCH --mem=0
#SBATCH --output=./ray-%j.out
#SBATCH --error=./ray-%j.err

set -eo pipefail

# 1) Modules / Conda
source /etc/profile.d/modules.sh
module purge
module load cuda/12.6 miniconda/24.7.1-py312
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
module load cudnn/9.6.0
module load nccl/2.24.3
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate llmbench_origin

# Ray temp をユーザ配下へ
export RAY_TMPDIR=/home/Competition2025/P07/shareP07/cache/
mkdir -p "$RAY_TMPDIR"; chmod 700 "$RAY_TMPDIR"
export TMPDIR="$RAY_TMPDIR"

# 2) Network / NCCL
export NCCL_DEBUG=TRACE
export GPU_MAX_HW_QUEUES=2
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_CHECKS_DISABLE=1
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9
export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=0
export NCCL_PROTO=Simple
export RCCL_MSCCL_ENABLE=0
export TOKENIZERS_PARALLELISM=false
export HSA_NO_SCRATCH_RECLAIM=1
export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK
unset ROCR_VISIBLE_DEVICES
ulimit -v unlimited

# 3) Topology
nodes_array=($(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' '))
head_node=${nodes_array[0]}
port=37173
dashboard_port=$((port + 1))

head_node_ip=$(srun -N1 -n1 -w "$head_node" hostname -I | awk '{print $1}')
if [[ "$head_node_ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"$head_node_ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then head_node_ip=${ADDR[1]}; else head_node_ip=${ADDR[0]}; fi
  echo "IPV6 detected. Use IPv4: $head_node_ip"
fi

ip_head=$head_node_ip:$port
export ip_head
echo "[INFO] Head IP → $ip_head"

# Ray head
srun -N1 -n1 -w "$head_node" bash -lc "\
source '$CONDA_BASE/etc/profile.d/conda.sh' && conda activate llmbench_origin && \
export RAY_TMPDIR=/tmp/r${SLURM_JOB_ID}; mkdir -p \$RAY_TMPDIR; chmod 700 \$RAY_TMPDIR; export TMPDIR=\$RAY_TMPDIR; \
ray start --head \
  --node-ip-address=$head_node_ip \
  --port=$port \
  --dashboard-port=$dashboard_port --dashboard-host=0.0.0.0 \
  --temp-dir=\$RAY_TMPDIR \
  --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=$SLURM_GPUS_PER_NODE --block" &
sleep 20

# Ray workers
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "[INFO] Launching worker on $node_i ..."
  srun -N1 -n1 -w "$node_i" bash -lc "\
    source '$CONDA_BASE/etc/profile.d/conda.sh' && conda activate llmbench_origin && \
    export RAY_TMPDIR=/tmp/r${SLURM_JOB_ID}; mkdir -p \$RAY_TMPDIR; chmod 700 \$RAY_TMPDIR; export TMPDIR=\$RAY_TMPDIR; \
    ip=\$(hostname -I | awk '{print \$1}'); \
    ray start --address $ip_head --node-ip-address=\$ip \
      --temp-dir=\$RAY_TMPDIR \
      --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=$SLURM_GPUS_PER_NODE --block" &
  sleep 5
done

# 6) Connectivity test（同環境で実行）
srun --overlap -N1 -n1 -c1 --gpus=0 -w "$head_node" bash -lc "\
source '$CONDA_BASE/etc/profile.d/conda.sh' && conda activate llmbench_origin && \
python - <<'PY'
import ray, json
ray.init(address='$ip_head')
print(json.dumps({'nodes': len(ray.nodes()),
                  'detail': [{'host': n['NodeManagerHostname'], 'alive': n['Alive']} for n in ray.nodes()]},
                 indent=2))
ray.shutdown()
PY"

# 7) Watchdog
ray_health_url="http://${head_node_ip}:${dashboard_port}/api/gcs_healthz"
ray_pids=($(jobs -pr))
echo "[INFO] Waiting on Ray daemons: ${ray_pids[*]}"

health_check () { curl -sf --max-time 10 "$ray_health_url" >/dev/null; }

while true; do
  for pid in "${ray_pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[ERROR] Ray process $pid exited."
      exit 1
    fi
  done
  if ! health_check; then
    echo "[ERROR] Ray dashboard unhealthy."
    exit 1
  fi
  sleep 300
done
