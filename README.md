# marl-copa
This is the implementation for [Coach-Player Multi-Agent Reinforcement Learning
for Dynamic Team Composition](https://arxiv.org/pdf/2105.08692.pdf) (ICML 2021)

## 1. Install the multiagent-particle-envs
```
pip install -e multiagent-particle-envs/
```

## 2. Run the experiments
```
./run.sh
```

## 4. Citations
Please consider citing [this paper](https://arxiv.org/pdf/2105.08692.pdf):
```
@InProceedings{pmlr-v139-liu21m,
  title = 	 {Coach-Player Multi-agent Reinforcement Learning for Dynamic Team Composition},
  author =       {Liu, Bo and Liu, Qiang and Stone, Peter and Garg, Animesh and Zhu, Yuke and Anandkumar, Anima},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {6860--6870},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/liu21m/liu21m.pdf},
  url = 	 {https://proceedings.mlr.press/v139/liu21m.html},
  abstract = 	 {In real-world multi-agent systems, agents with different capabilities may join or leave without altering the team’s overarching goals. Coordinating teams with such dynamic composition is challenging: the optimal team strategy varies with the composition. We propose COPA, a coach-player framework to tackle this problem. We assume the coach has a global view of the environment and coordinates the players, who only have partial views, by distributing individual strategies. Specifically, we 1) adopt the attention mechanism for both the coach and the players; 2) propose a variational objective to regularize learning; and 3) design an adaptive communication method to let the coach decide when to communicate with the players. We validate our methods on a resource collection task, a rescue game, and the StarCraft micromanagement tasks. We demonstrate zero-shot generalization to new team compositions. Our method achieves comparable or better performance than the setting where all players have a full view of the environment. Moreover, we see that the performance remains high even when the coach communicates as little as 13% of the time using the adaptive communication strategy.}
}

```
