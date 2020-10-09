# Report

## Implementation

## Learning Algorithm

## Summary of Hyperparameters
This project uses a number of experiences given by BATCH_SIZE = 64.

The parameters of the Actor and the Critic are updated by the soft-update with a fraction given by TAU = 0.01

Here are the parameters of the learning periodicity and the repetition: UPDATE_EVERY = 10 and N_LEARNING = 4

Learning rates for the Actor and Critic are given by LR_ACTOR = 5e-4 and LR_CRITIC = 5e-4, respectively.

Here are parameters of Ornstein–Uhlenbeck noise process: THETA = 0.01, SIGMA = 0.005, DECAY_FACTOR_S = 0.999, and DECAY_FACTOR_S = 0.999


## Architectures for the Actor and the Critic

## Results


![Figure of Score](https://github.com/hurxx018/Udacity_CollaborationAndCompetition/blob/master/scores_collaboration_and_competition.png)



## Future Work