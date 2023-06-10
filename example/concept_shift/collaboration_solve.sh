cd ../../src || exit

dataset='coarse-cifar100'
num_client=20
partition='stratified'    # Unbalanced label distribution
subsample='clusterexp_2_3.0' # Unbalanced quantities
data_seed=0

divergence='C'
divergence_seed=0

partition_config="client_${num_client}_partition_${partition}_subsample_${subsample}_seed_${data_seed}"


divergence_config="${divergence}_seed_${divergence_seed}"

C=8.0

python collab_opt.py \
  --dataset ${dataset} \
  --partition_config ${partition_config} \
  --divergence_config ${divergence_config} \
  --C ${C} \
  --seed 0
