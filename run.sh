#!/bin/bash

rm -rf sim0_h3
mkdir sim0_h3
mkdir sim0_h3/cpo
rm -rf sim0_h6
mkdir sim0_h6
mkdir sim0_h6/cpo

for t in {0..4}
do
  CUDA_VISIBLE_DEVICES="0" python cpo_main.py --logfile sim0_h3/cpo/trial_$t.out --seed=$t &
  p0=$!
  CUDA_VISIBLE_DEVICES="1" python cpo_main.py --logfile sim0_h6/cpo/trial_$t.out --max_episode_len=6 --threshold=-2.5 --seed=$t &
  p1=$!
  wait $p0
  wait $p1
done
