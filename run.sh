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

# define log output file
log_file="./logs/file_${AGGREGATOR}_${N_CLIENTS}_${BINARY}.log"
output_file="./result.txt"

echo "Starting server"

# run server
python server.py --n-clients $N_CLIENTS --binary $BINARY --aggregator "FedAvg"&
# python server.py --n-clients $N_CLIENTS --binary $BINARY --aggregator "FedAvg" &> $log_file&

sleep 10  # Sleep for 10s to give the server enough time to start and download the dataset

for i in $(seq 0 $((N_CLIENTS-1))); do
    echo "Starting client $i "
    python client.py --client-id=${i} --n-clients $N_CLIENTS --binary false &> /dev/null&
    echo "========================"
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
