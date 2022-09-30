#!/bin/bash

DATA=~/workdev/datasets/kits/kits19/data
OUT_DATA=~/workdev/datasets/kits/data


python preprocess_dataset.py --data_dir $DATA --results_dir $OUT_DATA
