# Baby Berry RL Environment

This repo contains an environment where a baby is placed in a grid world with a number of berries. The baby may move north, south, east or west to catch berries and eat them. Once all the berries are eaten the game is over and the baby experiences a large reward. The berries may be given random movement probabilities. This project has an associated blog post [here](https://sjhatfield.github.io/RL-Baby-Berry-Environment/).

Additionally, there may be a dad who moves around the world trying to pick up the baby and stop them eating all the tasty berries.

Here is an example of the baby moving randomly around the world. The baby is light blue, the berries are pink and the dumb (moves randomly with probability 50%) dad is in green.

![Baby moving randomly against dumb dad](https://github.com/sjhatfield/babyberry/blob/main/images/random-dumb.gif)

Here is the baby moving randomly with a smart dad who moves towards the baby with probability 50%.

![Baby moving randomly against smart dad](https://github.com/sjhatfield/babyberry/blob/main/images/random-smart.gif)

The purpose of this project was to use a variety of reinforcemnt learning algorithms in a new environment. In the models folder you can find the learning algorithm implementations. With the way the rewards were chosen, averaging 0 total reward over 200 consecutive epsiodes was considered *winning* the game against the **dumb dad**. Against the **smart dad** with a movement probability of 25%, averaging over -30 total reward was considered *winning*.

The algorithms were used against a dumb dad who moved randomly 50% of the time and a smart dad who moved towards the baby 25% of the time.

The best performing algorithm against the **dumb dad** was the double Q-learner for which a plot is shown below.

![Plot of average total reward over previous 200 episodes](https://github.com/sjhatfield/babyberry/blob/main/images/dumb_dad/double_Qlearner/episode_rewards.png?raw=true)

All learners were given the same hyperparameters to assess their performance fairly. The learners were given more episodes to train against the smart dad, 100,000 as opposed to 30,000. Epsilon was decayed over 90% of the episodes available for training, beginning at 1 and ending at 0.01. In reality, some of the learners may have completed the task quicker with a lower proportion of episodes to decay epsilon over but for a fair comparison this was left constant.

Here is the double Q-learner following its policy after training.

![Double Q-learner against dumb dad](https://github.com/sjhatfield/babyberry/blob/main/videos/double-Qlearner-dumb.gif)

The best performing algorithm against the **smart dad** was the Q-learner whose graph and an example of playback are below.

![Plot of average total reward over previous 200 episodes](https://github.com/sjhatfield/babyberry/blob/main/images/smart_dad/Qlearner/episode_rewards.png?raw=true)

![name-of-you-image](https://github.com/sjhatfield/babyberry/blob/main/videos/Qlearner-smart.gif)

# TODO

Implement n-step Tree Backup, off-policy Monte-Carlo control, Monte-Carlo Exploring Starts and Q(sigma) learning algorithms.

Try increasing the state view for the baby to 7.

Add random blocks of wall to the environment.

Change the action selection from "N", "S", "E", "W", "R" to 0, 1, 2, 3, 4

Experiment with adding a "stay still" action
