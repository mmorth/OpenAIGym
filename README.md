# OpenAIGym
This repository contains code for my OpenAIGym projects.

# Acrobot-v1 Deep Double Q-Learning
For the final project in my Deep Learning class (EE 526X), I created an implementation of the Deep Double Q-Learning algorithm for the Acrobot-v1 OpenAI Gym environment. I used neural networks for function approximation for the Q(S, A) values. Additionally, I used experience replay to update the Q function approximation neural networks.
The Acrobot-v1 environment consists of two links and two joints and the goal of the environment is to have the lower link clear the black bar, which is at a distance of one link from the top joint. My implementation allows the agent to clear the bar after less 200 steps. It takes about 10-15 episodes to achieve this performance. More specific implementation information can be found in my code above.
