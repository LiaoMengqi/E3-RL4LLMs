# E3-RL4LLMs

## Introduction
We enhance the training efficiency of RL by introducing a dynamic rollout 
budget allocation mechanism. 
For simple questions that the model can answer proficiently, 
we reduce their rollout budget, as performing reinforcement learning on such 
problems yields minimal gains. The saved rollout budget is reallocated to 
more challenging problems, thereby increasing the likelihood of 
obtaining correct answers. 

Additionally, to promote exploration without introducing harmful gradients, 
we propose a temperature scheduler that dynamically adjusts the temperature to maintain 
a stable policy entropy level, thereby enabling more extensive exploration during
training.  An annealing mechanism is further integrated to effectively balance 
exploration and exploitation.


## Train
requirements:
```text
verl==0.2.0
vllm==0.8.0
```

train
```shell
bash ./scripts/train.sh
```

