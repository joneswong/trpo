#!/bin/bash

#rm -rf sim0_h6
#mkdir sim0_h6
#rm -rf sim4_h6
#mkdir sim4_h6
rm -rf sim7_h6
mkdir sim7_h6
#rm -rf sim10_h6
#mkdir sim10_h6
#
#rm -rf sim0_h6/cpo
#mkdir sim0_h6/cpo
#rm -rf sim4_h6/cpo
#mkdir sim4_h6/cpo
rm -rf sim7_h6/cpo
mkdir sim7_h6/cpo
#rm -rf sim10_h6/cpo
#mkdir sim10_h6/cpo

for t in {0..4}
do
  #CUDA_VISIBLE_DEVICES="0" python cpo_main.py --logfile sim0_h6/cpo/trial_$t.out --max_episode_len=6 --threshold=-2.5 --seed=$t &
  #p0=$!
  #CUDA_VISIBLE_DEVICES="1" python cpo_main.py --logfile sim4_h6/cpo/trial_$t.out --max_episode_len=6 --threshold=-0.8 --level=4 --seed=$t &
  #p1=$!
  CUDA_VISIBLE_DEVICES="2" python cpo_main.py --logfile sim7_h6/cpo/trial_$t.out --max_episode_len=6 --threshold=-0.7 --level=7 --seed=$t &
  p2=$!
  #CUDA_VISIBLE_DEVICES="3" python cpo_main.py --logfile sim10_h6/cpo/trial_$t.out --num_pvs=10000 --threshold=-0.95 --level=10 --seed=$t &
  #p3=$!
  #wait $p0
  #wait $p1
  wait $p2
  #wait $p3
done
