# Udacity Project: Collaboration and Competition

## Problem
In the Tennis environment, two agents control rackets to bounce a ball over a net.  If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bound, it receives a reward of -0.01.  The goal of each agent is to keep the ball in play.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 over 100 consecutive episodes, after taking the maximum over both agents. Specifically,

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.

* This yields a single score for each episode.

* The average over 100 consecutive episodes is calculated. 

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

The observation space includes 8 variables consisting of the position (x, y) and velocity (v_x, v_y) of the ball and racket.  Each agent receives its own, local observation.  Three consecutive observations are an input state for the agents (The total number of entries in a state is 48 (= 8 x 3 x 2)), where 24 entries are for each agent.  An action of each agent is a vector of two continuous variables that correspond to movements toward (or away from) the net, and jumping.  Every entry in the action vector should be a value between -1 and 1.

## Dependencies
Set up your python environment by following the instruction
in [Deep Reinforcement Learning Nanodegree](https://github.com/udacity/deep-reinforcement-learning#dependencies)

Install the Tennis environment by following the instruction in [Here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet)


## Instructions
The agent utilizing the multi-agent Deep Deterministic Policy Gradient (maDDPG) algorithm is defined in ddpg_agent.py. 

The Actor and the Critic networks used in our maDDPG agents are defined in ddpg_model.py. 

The agent is trained by the following:

* python ./main_train_v1.py

To visualize the trained agent, run the following:

* python ./visualize_agent_v1.py

## Report
[Here is a report](Report.md)