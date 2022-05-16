#!/bin/bash


# lr=('0.01' '0.005' '0.001' '0.0005')
# reg_weight=('0.01' '0.001' '0.0001')
emb_size=(64)
lr=('0.01')
reg_weight=('1e-3')

dataset=('tmall' 'beibei')

for name in ${dataset[@]}
do
    for l in ${lr[@]}
    do
        for reg in ${reg_weight[@]}
        do
            for emb in ${emb_size[@]}
            do
                echo 'start train: '$name
                `
                    python main.py \
                        --lr ${l} \
                        --reg_weight ${reg} \
                        --data_name $name \
                        --embedding_size $emb
                `
                echo 'train end: '$name
            done
        done
    done
done