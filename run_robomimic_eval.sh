#!/bin/bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate pi0
source ~/openpi/third_party/robomimic/examples/robomimic/.venv/bin/activate

# 设置需要运行的 suite / task 和对应 GPU 编号
# robomimic 这里 suite 就可以理解为 task 名（如 lift / can / square / tool_hang）

TASKS=("lift" "can" "square")
# TASKS=("transport")
GPUS=(1 2 3)
# GPUS=(4 5 6)
# GPUS=(0 1 2 3 4 5)
PORTS=(8000 8001 8002)
# PORTS=(8003 8004 8005)
# PORTS=(8001)
NUM_TRIALS=50
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3

for i in "${!TASKS[@]}"
do
  GPU=${GPUS[$i]}
  TASK=${TASKS[$i]}
  PORT=${PORTS[$i]}

  echo ">>> [GPU $GPU] 启动 serve_policy.py 和 robomimic 任务: $TASK (num_trials_per_task=$NUM_TRIALS)"

  (
    # 启动 serve_policy.py（后台）
    CUDA_VISIBLE_DEVICES=$GPU uv run scripts/serve_policy.py --env ROBOMIMIC --port $PORT > logs/server_${TASK}_lora_2w.log 2>&1 &
    SERVER_PID=$!

    # 等待 server 启动
    sleep 3

    # 启动 robomimic websocket 评测（前台）
    export PYTHONPATH=$PYTHONPATH:$PWD/third_party/robomimic
    CUDA_VISIBLE_DEVICES=$GPU python examples/robomimic/main.py \
        --args.tasks "$TASK" \
        --args.num-trials-per-task $NUM_TRIALS \
        --args.port $PORT

    # 关闭后台 serve_policy
    kill $SERVER_PID
  ) &
done

wait
echo "✅ 所有 Robomimic 任务完成。"