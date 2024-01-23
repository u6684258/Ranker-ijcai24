# Guiding GBFS through Learned Pairwise Rankings

This repository contains source code for the paper [Guiding GBFS through Learned Pairwise Rankings](https://github.com/u6684258/Ranker-ijcai24). The implementation is based on [**GOOSE**: **G**raphs **O**ptimised f**O**r **S**earch **E**valuation](https://github.com/DillonZChen/goose). 

## Repository Structure

```
|-benchmarks
  |-ipc2023-learning-benchmarks
|-learner
  |-dataset
  |-gnn
  |-hgn
  |-models
  |-scripts
  |--train_rank.py
|-apptainer.goose.exp
|-apptainer.hgn.exp
```
- The `benchmark` folder contains planning problems from the IPC23-learning track.
- The `learner` folder contains the main experiment scripts. 
  - `train_rank.py`: The training script entry point.   
  - `dataset`: Scripts to generate training data and cache them under `root/data/`.
  - `gnn` & `hgn`: Implementation of training/validation steps.
  - `models`: Implementation of gnn and hgn models.
  - `scripts`: Full experiment scripts (train + test).
- The `planner` folder contains planners that use the learnt configurations to solve problems.
- The `apptainer.*.exp` are definition files for building containers. 

## Running Experiments

1. Build the corresponding container, e.g. :
  ```
  singularity build experiment.sif apptainer.goose.exp
  ```
2. Run the container, e.g.:
  ```
  singularity run -B <path_to_repo>/data/:/data -B <path_to_repo>/logs/:/logs <path_to_repo>/experiment.sif <aggr> <nlayer> <encode> <model> <domain> <train-only>
  ```
  The container file takes 6 inputs:
  - "aggr": The aggregation function for GOOSE. HGN ignores this parameter, however, for consistency HGN also requires this input as a placeholder. 
  - "nlayer": number of message passing layers in the GNN. 
  - "encode": Input encoding method. We used 'llg' in our experiments for GOOSE. HGN ignores this parameter.
  - "model": The neural network model. Choose from:
    - \*, for the original regression model
    - \*-rank, for the Ranking model
    - \*-loss, for the original regression model with [ranking based loss function](https://arxiv.org/abs/2310.19463).
    - "\*" can be either "gnn" for GOOSE or "hgn" for HGN. 
  - "train-only": Set 1 to skip the testing phases. Set 0 otherwise. 
Example command:
  ```
  singularity run -B <path_to_repo>/data/:/data -B <path_to_repo>/logs/:/logs <path_to_repo>/experiment.sif mean 4 llg gnn-rank blocksworld 0
  ```

