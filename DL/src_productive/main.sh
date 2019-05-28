#!/bin/bash

for input in `ls ../dl_input/*.pkl`; do
        #statements
        echo $input
        DATA=$(basename "$input" "_Grid.pkl")
        echo $DATA
        python main_5.py -i $input \
                                 -o ../../DL_output/${DATA}
        python main_purine.py -i ../../DL_output/${DATA}_result.pickle \
                                 -o ../../DL_output/${DATA}
        python main_pyrimidine.py -i ../../DL_output/${DATA}_result.pickle \
                                 -o ../../DL_output/${DATA}
        python final_format.py -i ../../DL_output/${DATA}_result.pickle \
                                 -o ../../DL_output/${DATA}
done

