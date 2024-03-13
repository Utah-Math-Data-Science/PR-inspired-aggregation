#!/bin/bash

<<COMMENT
Generate Sweeps
================
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

<<COMMENT
Generating Sweeps by Folder
---------------------------
COMMENT

function binary_tuning {
    direct=/root/workspace/PR-inspired-aggregation/sweeps/binary_tuning/
    echo -e "\n----\nBINARY TUNING\n----" &>> /root/workspace/PR-inspired-aggregation/sweeps/sweep_ids.log
    for file in $direct/*; do
        wandb sweep --project pr-inspired-aggregation $file --name ${file:62:-5} &>> /root/workspace/PR-inspired-aggregation/sweeps/sweep_ids.log
        sleep 0.5
    done
}

function binary_ablation {
    direct=/root/workspace/PR-inspired-aggregation/sweeps/binary_ablation/
    echo -e "\n----\nBINARY ABLATION\n----" &>> /root/workspace/PR-inspired-aggregation/sweeps/sweep_ids.log
    for file in $direct/*; do
        wandb sweep --project pr-inspired-aggregation $file --name ${file:64:-5} &>> /root/workspace/PR-inspired-aggregation/sweeps/sweep_ids.log
        sleep 0.5
    done
}

function multi_tuning {
    direct=/root/workspace/PR-inspired-aggregation/sweeps/multi_tuning/
    echo -e "\n----\nMULTI TUNING\n----" &>> /root/workspace/PR-inspired-aggregation/sweeps/sweep_ids.log
    for file in $direct/*; do
        wandb sweep --project pr-inspired-aggregation $file --name ${file:61:-5} &>> /root/workspace/PR-inspired-aggregation/sweeps/sweep_ids.log
        sleep 0.5
    done
}

function multi_ablation {
    direct=/root/workspace/PR-inspired-aggregation/sweeps/multi_ablation/
    echo -e "\n----\nMULTI ABLATION\n----" &>> /root/workspace/PR-inspired-aggregation/sweeps/sweep_ids.log
    for file in $direct/*; do
        wandb sweep --project pr-inspired-aggregation $file --name ${file:63:-5} &>> /root/workspace/PR-inspired-aggregation/sweeps/sweep_ids.log
        sleep 0.5
    done
}


function todo {
    direct=/root/workspace/PR-inspired-aggregation/sweeps/todo/
    echo -e "\n----\nTODO\n----" &>> /root/workspace/PR-inspired-aggregation/sweeps/sweep_ids.log
    for file in $direct/*; do
        wandb sweep --project pr-inspired-aggregation $file --name ${file:53:-5} &>> /root/workspace/PR-inspired-aggregation/sweeps/sweep_ids.log
        sleep 0.5
    done
}

#----------------------------------------------------------------------------------------------------------------------------------------------------

if [[ -z "$1" ]]; then
  echo -e "To generate sweeps, please specify the sweeps directory\n\tbash generate.sh <sweep-dir>"
else
  $1
fi