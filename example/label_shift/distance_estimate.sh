cd ../../src || exit

gpu=0

dataset='fmnist'
num_client=20
partition='cluster4_0.25'    # Unbalanced label distribution
subsample='clusterexp_2_5.0' # Unbalanced quantities
seed=0

partition_config="client_${num_client}_partition_${partition}_subsample_${subsample}_seed_${seed}"
model='mlp'

CUDA_VISIBLE_DEVICES=${gpu} python disc.py \
  --dataset ${dataset} \
  --partition_config ${partition_config} \
  --divergence C \
  --use_valid \
  --model ${model} \
  --lr 0.01 \
  --rounds 2000 \
  --eval_rounds 10 \
  --early_stop 10 \
  --use_valid \
  --seed 0 \
  --cuda
