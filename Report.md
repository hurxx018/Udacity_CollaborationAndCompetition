# Report

## Implementation
To solve the environment, I employed mutli-agent deep deterministic policy-gradient (DDPG) algorithm.  The environment includes two rackets of which each is represented by a local actor and critic and a target actor and critic. The learning agent works with the eight networks in total. An input of the actor of each racket is its own observation and the actor output has two continuous variables whose values are between -1 and 1.  The critic uses the observations and the actions of the two rackets as the input.  The local actors and critics were optimized by using the Adam. The agent was trained with 6000 episodes.

The replay buffer was used that has a size of 500000 experiences.  The rewards of each batch are normalized before the learning stage.  The agents learned 6 times every two experiences (UPDATE_EVERY = 2 and N_LEARNING = 6).  The target networks were soft-updated.  

## Summary of Hyperparameters
This project uses for the agent's learning a number of experiences given by BATCH_SIZE = 32.

The parameters of the Actor and the Critic are updated by the soft-update with a fraction given by TAU = 0.02

Here are the parameters of the learning periodicity and the repetition: UPDATE_EVERY = 2 and N_LEARNING = 6

Learning rates for the Actor and Critic are given by LR_ACTOR = 7e-5 and LR_CRITIC = 7e-5, respectively.

Here are parameters of Ornstein–Uhlenbeck noise process: THETA = 0.01, SIGMA = 0.005, DECAY_FACTOR_S = 0.999, and DECAY_FACTOR_S = 0.999


## Architectures for the Actor and the Critic
The Actor network consists of an input layer, two hidden layers (FC1 and FC2), and one output layer.  The FC1 and FC2 layers include 256 and 512 nodes, respectively.  They are used with leakyReLU activation function with a negative slope of 0.2.  Batch normalization (BN1) is applied to the output of the layer FC1.  The activation function of the output layer is hyperbolic-tangent that ensures the output values of the range from -1 to 1.

The Critic network includes an input layer, three hidden layers (FCS1, FC2, FC3), and an output layer.  The state space of 48 variables is converted to a layer of 256 nodes with the LeakyRelu activation with a negative slope of 0.2.  The output of the input layer is concatenated with the action values of the two rackets (2x2), that is denoted by FCS1. The next two hidden layers were used to calculate the state-action value with the LeakyRelu activatoin.  The layer FC2 has 512 nodes and the FC3 includes 256 nodes.  The output is a linear logit.

## Results
![Figure of Score](https://github.com/hurxx018/Udacity_CollaborationAndCompetition/blob/master/scores_collaboration_and_competition.png)

The agent was trained with 6000 episodes. The score of each episode is shown by a blue line and the mean of 100 consecutive scores is shown by an orange line.  The initial mean score was close to 0, where the agent does not know the environment.  The scores of 0.1 indicates that one of the two rockets learned to pass a ball. As the two rockets learns to pass a ball, the mean score stabilizes to be around 0.1 from 1000 to 3000 episodes.  After this stage, two rockets learn to pass a ball to each other several times.  The mean score was measured to be above 0.5 around 3500 episodes where some of scores were close to ~2.6.  However, the agent lost the capability of continuing to pass a ball from one racket to the other.  The capability was slowly improved during the later episodes but the mean scores were not as high as the score of 0.5.


## Future Work
Two rackets are learning together in this environment while they are interacting with each other.  The optimal point in the loss function for the agent seems to be very narrow that makes the training difficult because hyper-parameters should be identified well.  The next step is to relieve this type of difficulty.