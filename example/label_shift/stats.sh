cd ../../src || exit

dataset='fmnist'
algorithm='fedavg'

{
  echo "Local Training:" &&
    python stats.py \
      --history_path "../history/${dataset}/${algorithm}_local.pkl"
} && {
  echo "Global Training:" &&
    python stats.py \
      --history_path "../history/${dataset}/${algorithm}_global.pkl" \
      --ref_history_path "../history/${dataset}/${algorithm}_local.pkl" \
      --ref
} && {
  echo "FedCollab:" &&
    python stats.py \
      --history_path "../history/${dataset}/${algorithm}_fedcollab.pkl" \
      --ref_history_path "../history/${dataset}/${algorithm}_local.pkl" \
      --ref
}
