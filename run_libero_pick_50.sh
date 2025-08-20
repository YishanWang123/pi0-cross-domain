#!/bin/bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate pi0
source examples/libero/.venv/bin/activate
# 或者用 uv，但不要两者一起用
# export XLA_PYTHON_CLIENT_MEM_FRACTION 可保留
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3

SUITES=("libero_object")   # 先跑一个套件验证也行
GPUS=(0)
PORTS=(8000)

NUM_TRIALS=50

# 你挑出来的 id 列表（按需改）
TASKS_libero_spatial=(2 4 9)
TASKS_libero_object=(6 8 9)
TASKS_libero_goal=(4 8 9)
TASKS_libero_10=(2 6 7 9)
TASKS_libero_90=(2 8 9 12 14 15 16 18 19 21 23 26 27 30 32 34 41 48 50 52 53 54 56 58 59)

for i in "${!SUITES[@]}"; do
  SUITE=${SUITES[$i]}
  GPU=${GPUS[$i]}
  PORT=${PORTS[$i]}
  varname="TASKS_${SUITE}"
  declare -n ref="$varname"   # ref 现在就是 $varname 指向的数组
  IDS=("${ref[@]}")
  echo "SUITE=$SUITE"
  echo "varname=$varname"
  declare -p "$varname"

  echo ">>> [GPU $GPU] suite=$SUITE ids=(${IDS[*]}) trials=$NUM_TRIALS"
  echo "IDS=(${IDS[@]})"

  (
    CUDA_VISIBLE_DEVICES=$GPU uv run scripts/serve_policy.py --env LIBERO --port $PORT > "logs/server_${SUITE}.log" 2>&1 &
    SERVER_PID=$!
    sleep 3

    export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
    CUDA_VISIBLE_DEVICES=$GPU python examples/libero/main.py \
      --args.task_suite_name "$SUITE" \
      --args.num_trials_per_task $NUM_TRIALS \
      --args.port $PORT \
      --args.only_task_ids "${IDS[@]}"

    kill $SERVER_PID || true
  ) &
done

wait
echo "✅ done."