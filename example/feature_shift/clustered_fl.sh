cd ../../src || exit

gpu=1

dataset='rotated-cifar10'
num_client=20
partition='stratified'       # Balanced label distribution
subsample='clusterexp_2_2.0' # Unbalanced quantities
data_seed=0

divergence='C'
divergence_seed=0

C=10.0
solver_seed=0

partition_config="client_${num_client}_partition_${partition}_subsample_${subsample}_seed_${data_seed}"
divergence_config="${divergence}_seed_${divergence_seed}"
collab_config="${divergence_config}_discrete_C_${C}_seed_${solver_seed}"

collabs=('global' 'local' "${collab_config}")
shortcuts=('global' 'local' 'fedcollab')


model='cnn'
lr=0.01
algorithm='fedavg'
seed=0

for i in {0..2}; do
  {
    CUDA_VISIBLE_DEVICES=${gpu} python main.py \
      --dataset ${dataset} \
      --partition_config ${partition_config} \
      --collab_config "${collabs[i]}" \
      --model ${model} \
      --algorithm ${algorithm} \
      --gm_rounds 500 \
      --lm_opt sgd \
      --lm_lr ${lr} \
      --lm_epochs 1 \
      --seed ${seed} \
      --cuda \
      --history_path "../history/${dataset}/${algorithm}_${shortcuts[i]}.pkl"
  } &
done
