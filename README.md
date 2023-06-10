# FedCollab

This is the official implementation of the following paper:

Wenxuan Bao, Haohan Wang, Jun Wu, Jingrui He. _Optimizing the Collaboration Structure in Cross-Silo Federated Learning_.
ICML 2023.

## Introduction

**FedCollab** is a federated learning algorithm which alleviate the negative transfer problem by by clustering clients
into
non-overlapping coalitions based on clients' data quantities and their pairwise distribution distances.

## Requirements

- python 3.8.5
- cudatoolkit 10.2.89
- cudnn 7.6.5
- pytorch 1.11.0
- torchvision 0.12.0
- numpy 1.18.5
- tqdm 4.65.0
- matplotlib 3.7.1

## Run

Here we provide an example script for experiments with FedAvg.

1. Generate data partition

```shell
cd ./example/${setting}
./data_prepare.sh
```

`${setting}` should be filled with `label_shift`, `feature_shift` or `concept_shift`. The dataset and its partition will
be saved to `~/data`.

2. Estimate pairwise divergence between each pair of clients

```shell
./distance_estimate.sh
```

The estimated pairwise divergence will be saved to `./divergence`.

3. Solve for the optimal collaboration structure

```shell
./collaboration_solve.sh
```

The solved collaboration structure will be saved to `./collab`.

4. Run clustered federated learning

```shell
./clustered_fl.sh
```

The training history will be saved to `./history`.

5. Calculate metrics (Acc, IPR, RSD)

```shell
./stats.sh
```

The statistics will be printed.

## Expected outputs

The expected extimated pairwise divergence, collaboration structure, and training history are given in this repository.
We also list the expected accuracy (Acc), incentivized participation rate (IPR), and reward standard deviation (RSD)
below. _Notice that this is the result with one seed, while we showed the results from five difference random seeds in
our paper._

### Label shift (FashionMNIST)

| Method                |  Acc   |  IPR   |  RSD   |
|-----------------------|:------:|:------:|:------:|
| Local Train           | 0.8580 |   -    |   -    |
| FedAvg (Global Train) | 0.4674 | 0.4500 | 0.4076 |
| FedAvg + FedCollab    | 0.9247 | 1.0000 | 0.0626 |

### Feature shift (Rotated CIFAR-10)

| Method                |  Acc   |  IPR   |  RSD   |
|-----------------------|:------:|:------:|:------:|
| Local Train           | 0.3829 |   -    |   -    |
| FedAvg (Global Train) | 0.4447 | 0.9000 | 0.0408 |
| FedAvg + FedCollab    | 0.5286 | 1.0000 | 0.0413 |

### Label shift (Coarse CIFAR-100)

| Method                |  Acc   |  IPR   |  RSD   |
|-----------------------|:------:|:------:|:------:|
| Local Train           | 0.3032 |   -    |   -    |
| FedAvg (Global Train) | 0.2649 | 0.5000 | 0.1111 |
| FedAvg + FedCollab    | 0.4041 | 1.0000 | 0.0256 |

## Citation

If you find this paper or repository helpful in your research, please cite our paper:

```text
To be released by the conference. 
```