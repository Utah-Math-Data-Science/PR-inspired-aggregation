#!/bin/bash

<<COMMENT
Run Sweeps
===========
    - unfortunately there  isn't a good way to search for each layer variable so I wrote this to automate it.
    - generate the runs using ``generate.sh <sweep-dir>``
    - look up the sweep ids or copy them from the print out (sorry I didn't automate this)
    - put them in the sweep_ids list
    - use 'bash ogb-arxiv.sh run <device>' to iterate over all of them
    CAUTION: you want your sweeps to have a run_cap to use this.
COMMENT

wandb login
mkdir -p /root/workspace/out/
cd /root/workspace/out/

#----------------------------------------------------------------------------------------------------------------------------------------------------

if [[ -z "$1" ]]; then
  echo -e "To run sweeps, please specify the device\n\tbash run.sh <device#>"
else
	echo "running on device $1"
  sweep_ids=($(grep "wandb: Creating sweep with ID:" /root/workspace/PR-inspired-aggregation/sweeps/sweep_ids.log | cut -d " " -f 6))
  echo "$sweep_ids"
  for sweep_id in "${sweep_ids[@]}"; do
      echo "Running: $sweep_id on device $1"
      CUDA_VISIBLE_DEVICES=$1 wandb agent utah-math-data-science/PR-inspired-aggregation/$sweep_id
  done
fi
