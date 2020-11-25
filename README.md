# Baby Berry RL Environment

This repo contains an environment where a baby is placed in a grid world with a number of berries. The baby may move North, South, East or West to catch berries and eat them. Once all the berries are eaten the game is over and the baby experiences a large reward. The berries may be given random movement probabilities.

Additionally, there may be a dad who moves around the world trying to pick up the baby and stop them eating all the tasty berries.

Here is an example of the baby moving randomly around the world. The baby is light blue, the berries are pink and the dumb (moves randomly) dad is in green.

The purpose of this project was to use a variety of Reinforcement Learning algorithms in a new environment. In the models folder you can find the learning algorithm implementations. With the way the rewards were chosen, averaging over 20 total reward over 200 consecutive epsiodes was considered *winning* the game.

The algorithms were used against a dumb dad who moved randomly 50% of the time and a smart dad who moved towards the baby 50% of the time.

The best performing algorithm against the **dumb dad** was the double Q-learner for which a plot is shown below.

![Plot of average total reward over previous 200 episodes](https://github.com/sjhatfield/babyberry/blob/main/images/dumb_dad/double_Qlearner/episode_rewards.png?raw=true)

The only parameter that was tuned was the proportion of the episodes available for training to decay epsilon over. Epsilon being the probability of a random action being chosen in the greedy action selection. For algorithms that converged quicker this could be set to something like 0.4 to converge as quickly as possible,