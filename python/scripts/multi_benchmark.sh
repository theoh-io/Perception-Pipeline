#!/bin/bash

END=20
for i in $(seq 1 $END)
    do
    name_file="person-${i}"
    echo "name of file: ${name_file}"
    python3 modif_cfg.py -id $i -v
    python3 -W ignore run.py 
    done
