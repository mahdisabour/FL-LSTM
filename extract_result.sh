#!/bin/bash

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

N_CLIENTS=$N_CLIENTS
AGGREGATOR=$AGGREGATOR
BINARY=$BINARY

log_file="./logs/file_${AGGREGATOR}_${N_CLIENTS}_${BINARY}.log"
output_file="./result.txt"

echo $log_file >> $output_file
tail -n 22 $log_file >> $output_file
echo "+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=" >> $output_file
