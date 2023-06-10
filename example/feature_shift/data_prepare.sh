cd ../../src || exit

gpu=1

dataset='cifar10'
num_client=20
partition='stratified'       # Unbalanced label distribution
subsample='clusterexp_2_2.0' # Unbalanced quantities
seed=0

partition_config="client_${num_client}_partition_${partition}_subsample_${subsample}_seed_${seed}"
rotation_config='20client_50'  # 20 clients

{
  python ./data_partition.py \
    --dataset ${dataset} \
    --num_client ${num_client} \
    --partition ${partition} \
    --subsample ${subsample} \
    --seed ${seed}
} && {
  CUDA_VISIBLE_DEVICES=${gpu} python ./rotation_prepare.py \
    --dataset ${dataset} \
    --partition_config ${partition_config} \
    --rotation_config ${rotation_config}
}
