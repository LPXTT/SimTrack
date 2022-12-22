#NNODES=${NNODES:-1}
#NODE_RANK=${NODE_RANK:-0}
#PORT=${PORT:-29525}
#MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PYTHONPATH="$(dirname $0)":$PYTHONPATH

#srun -p ai4science -n1 --gres=gpu:8 --ntasks-per-node=1 --cpus-per-task=5 --quotatype spot \
python tracking/test.py simtrack baseline --dataset lasot --threads 32
