cd ../../src || exit

gpu=2

dataset='coarse-cifar100'
num_client=20
partition='stratified'    # Unbalanced label distribution
subsample='clusterexp_2_3.0' # Unbalanced quantities
seed=0

partition_config="client_${num_client}_partition_${partition}_subsample_${subsample}_seed_${seed}"
labelswap_config='20client_6'
model='mlpfe_freeze'

CUDA_VISIBLE_DEVICES=${gpu} python disc.py \
  --dataset ${dataset} \
  --partition_config ${partition_config} \
  --labelswap_config ${labelswap_config} \
  --divergence C \
  --use_valid \
  --model ${model} \
  --lr 0.05 \
  --rounds 2000 \
  --eval_rounds 10 \
  --early_stop 10 \
  --use_valid \
  --seed 0 \
  --cuda
