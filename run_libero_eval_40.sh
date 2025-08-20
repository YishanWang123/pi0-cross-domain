#!/bin/bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate pi0
source examples/libero/.venv/bin/activate
# 设置需要运行的 suite 和对应 GPU 编号
# SUITES=("libero_90")
SUITES=("libero_90")
GPUS=(0)
# GPUS=(1 2 3)
PORTS=(8000)
# PORTS=(8001)

NUM_TRIALS=1
# 设置显存使用上限
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3

# 遍历任务
for i in "${!SUITES[@]}"
do
  GPU=${GPUS[$i]}
  SUITE=${SUITES[$i]}
  PORT=${PORTS[$i]}

  echo ">>> [GPU $GPU] 启动 serve_policy.py 和 suite: $SUITE (num_trials_per_task=$NUM_TRIALS)"

  (
    # 启动 serve_policy.py（后台）
    CUDA_VISIBLE_DEVICES=$GPU uv run scripts/serve_policy.py --env LIBERO --port $PORT > logs/server_${SUITE}.log 2>&1 &
    SERVER_PID=$!

    # 等待 server 启动
    sleep 3

    # 启动 main.py（前台）
    export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
    CUDA_VISIBLE_DEVICES=$GPU python examples/libero/main.py --args.task_suite_name "$SUITE" --args.num_trials_per_task $NUM_TRIALS --args.port $PORT

    # 关闭后台 serve_policy
    kill $SERVER_PID
  ) &
done

wait
echo "✅ 所有 LIBERO 任务完成。"