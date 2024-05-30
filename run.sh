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
if [[ $BINARY == "false" ]]; then
    python server.py --n-clients $N_CLIENTS --aggregator $AGGREGATOR &> $log_file&
elif [[ $BINARY == "true" ]]; then
    python server.py --n-clients $N_CLIENTS --binary $BINARY --aggregator $AGGREGATOR &> $log_file&
fi

sleep 10  # Sleep for 10s to give the server enough time to start and download the dataset

for i in $(seq 0 $((N_CLIENTS-1))); do
    echo "Starting client $i "
    if [[ $BINARY == "false" ]]; then
        python client.py --client-id=${i} --n-clients $N_CLIENTS &> /dev/null&
    elif [[ $BINARY == "true" ]]; then
        python client.py --client-id=${i} --binary $BINARY --n-clients $N_CLIENTS &> /dev/null&
    echo "========================"
    fi
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
