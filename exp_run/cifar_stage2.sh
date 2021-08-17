#!/bin/bash

set -x

for sup in True #False
do
  for st1_epoch in 200 400 600
  do
    for st2_epoch in 1 #00 200
    do
      cuda_idx=2
      ip=10001
      for ratio in 0.1 #0.01
      do
        for bs in 128 #256 512
        do
          CUDA_VISIBLE_DEVICES=${cuda_idx} python train_cifar_stage2.py --dist-url tcp://localhost:${ip} --multiprocessing-distributed --world-size 1 --rank 0 --pretrained runs/cifar100_lt_${ratio}_${bs}_${st1_epoch}/stage1/checkpoint_last.pth.tar --workers 0 --batch-size ${bs} --epochs $st2_epoch --imb_ratio $ratio --supervised $sup > /dev/null &
          cuda_idx=$(expr $cuda_idx + 1)
          ip=$(expr $ip + 1)
        done
      done
      wait
    done
  done
done
