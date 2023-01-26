# Robust Meta Reinforcement Learning (RoML) with MAML

The paper [**Train Hard, Fight Easy: Robust Meta Reinforcement Learning**]() introduces RoML - a meta-algorithm that takes any meta-learning baseline algorithm and generates a robust version of it.
This repo implements RoML on top of the [MAML](https://arxiv.org/abs/1703.03400) algorithm for meta learning.

RoML was originally developed for reinforcement learning problems, but is demonstrated here to be applicable to supervised meta learning as well.
For the demonstration, the toy sine-regression problem is used.
To reproduce the experiments, just run the jupyter notebook in the *Sine* directory.

See [here](https://github.com/ido90/RobustMetaRL) more details about what is RoML and how to use it in general, as well as implementation on top of other baselines.
