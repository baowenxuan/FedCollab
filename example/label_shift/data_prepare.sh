cd ../../src || exit

dataset='fmnist'
num_client=20
partition='cluster4_0.25'    # Unbalanced label distribution
subsample='clusterexp_2_5.0' # Unbalanced quantities
seed=0

python ./data_partition.py \
  --dataset ${dataset} \
  --num_client ${num_client} \
  --partition ${partition} \
  --subsample ${subsample} \
  --seed ${seed}
