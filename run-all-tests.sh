#!/bin/bash

for aggregator in "FedAvg" "FedAdagrad" "FedYogi" "FedAvgM" "FedTrimmedAvg"
do
    for n_clients in 1 2 4 8
    do
        for type in true false
        do
            ./run.sh AGGREGATOR=$aggregator N_CLIENTS=$n_clients BINARY=$type
            ./extract_result.sh AGGREGATOR=$aggregator N_CLIENTS=$n_clients BINARY=$type
        done    
    done
done 