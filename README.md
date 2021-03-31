# Ballz-RL

## Overview
Reinforcement Learning Solution to the Ketchapp game: Ballz  
We wrote a custom environment of Ballz by OpenAI Gym and created an agent by ddpg. The ddpg implementation is referenced to [Deep Deterministic Policy Gradient on Pytorch](https://github.com/ghliu/pytorch-ddpg). The model is modified to fit Ballz environment. 

## Dependencies
- Python 3.7.9
- Pytorch 1.7.0
- OpenAI Gym

## Demo videos
|           | Train on same env | Train on random env |
| :-------------: | :---------------: | :---------------: |
| Simple    | [Link](https://drive.google.com/file/d/1Ulp3BowwyG5c6JU1Txhn-jttF4U7NjFn/view?usp=sharing) | [Link](https://drive.google.com/file/d/1Hzm_CxDCCX1LxNH73tqZi_MzNhN_nSl0/view?usp=sharing) | 
| Complex   | [Link](https://drive.google.com/file/d/1qsno8hEVt0ocgRO8umvBaZiRGK238qNS/view?usp=sharing) | [Link](https://drive.google.com/file/d/1qucm4wpVI0FqokhbMZajI4qGpE6oRVS0/view?usp=sharing) |

## Run
- Training
```
$ python main.py --mode train3 --debug --warmup 6000 --train_iter 26000 --max_episode_length 20 --agent 3 --noise OU --ou_sigma 0.5
```

- Testing
```
$ python main.py --debug --mode test3 --agent 3 --resume output/Ballz-run1/ --pattern 1
```
