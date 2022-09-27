#!/bin/bash


### Generate 50000 and 59999 and loops until that number isn't found in the list of used ports.
MPORT=1191  ## Using default port 56110
echo "checking if $MPORT used on ${MASTER} " 
CHECK=$(ss -tan | grep $MPORT)  ### if 56110 is not free
if [[ ! -z $CHECK  ]]; then
    echo "Default port ${MPORT} is used, rescan free ports... " 
    while [[ ! -z $CHECK ]]; do
       MPORT=$(( ( RANDOM % 50000 )  + 50000 ))
       CHECK=$(ss -tan | grep $MPORT)
    done
fi
echo " Using $MPORT " 
