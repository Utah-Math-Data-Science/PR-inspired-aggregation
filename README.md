# Douglas-Rachford Graph Neural Network

This repository is the official PyTorch implementation of "Monotone Operator Theory-Inspired Message Passing for Learning Long-Range Interaction on Graphs."


## Installation

**Dockerfile**: Recommended install using `Dockerfile.drgnn`.


**Manual**: Alternatively use `pip3 install -r requirements.txt` to manually install.


## Usage

**agg**: The `./agg/` directory contains all the relevant code for DRGNN.

**tasks**: Individual tasks are located in the `./tasks/` directory with `.py` driver files and `.yaml` settings files.

`python3 ./tasks/arxiv.py`

**sweeps**: Sweep results are run using [wandb](https://wandb.ai/) and can be easily instantiated and run by performing the following:

1. Move desired runs to `./sweeps/todo/`.

```
	mv ./sweeps/tables/* ./sweeps/todo/
```

2. Generate `sweep_ids.log`.

```
	bash ./sweeps/generate.sh todo
```

3. Run the sweep.

```
	bash ./sweeps/run.sh <device>
```

**analysis**: All other results are found in the `./analysis/` directory.

**baselines**: All baseline codes can be found in the `./baselines/` directory.


## Citation

Awaiting publication.
