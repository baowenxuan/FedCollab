cd ../../src || exit

dataset='coarse-cifar100'
num_client=20
partition='stratified'    # Unbalanced label distribution
subsample='clusterexp_2_3.0' # Unbalanced quantities
seed=0

python ./data_partition.py \
  --dataset ${dataset} \
  --num_client ${num_client} \
  --partition ${partition} \
  --subsample ${subsample} \
  --seed ${seed}
