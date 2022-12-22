PYTHONPATH="$(dirname $0)":$PYTHONPATH

#srun -p ai4science -n1 --gres=gpu:8 --ntasks-per-node=1 --cpus-per-task=5 --quotatype spot \
python tracking/train.py --script simtrack --config baseline_got10k_only --save_dir . --mode multiple --nproc_per_node 8
