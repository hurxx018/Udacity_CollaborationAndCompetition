from collections import deque
import time

from unityagents import UnityEnvironment

import numpy as np
import matplotlib.pyplot as plt
import torch

from ddpg_agent import Agent

def train_ddpg_v1(
    env,
    agent,
    n_episodes = 100
    ):

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size

    scores_deque = deque(maxlen = 100)
    scores = []
    mean_scores = []
    max_mean_score = 0.5
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode = True)[brain_name] # reset the environment
        states = env_info.vector_observations               # get the current state (for each agent)
        num_agents = len(env_info.agents)
        # scores_agent = np.zeros(num_agents)                          # initialize the score (for each agent)
        scores_agents = np.zeros(num_agents)
        agent.reset()
        while True:
            actions = agent.act(states, add_noise = True)
            # print(actions)
            env_info = env.step(actions)[brain_name] # send all actions to the environment

            next_states = env_info.vector_observations
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished

            agent.step(states, actions, rewards, next_states, dones)

            states = next_states
            scores_agents += rewards
            if np.any(dones): # exit loop if episode finished
                break

        score = np.max(scores_agents)
        scores_deque.append(score)
        scores.append(score)
        mean_scores.append(np.mean(scores_deque))
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, mean_scores[-1], score), end="")
        if i_episode % 100 == 0:
            # torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            # torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}\t'.format(i_episode, mean_scores[-1]))
        if max_mean_score < mean_scores[-1]:
            max_mean_score = mean_scores[-1]
            torch.save(agent.actor_local1.state_dict(), 'checkpoint_actor1.pth')
            torch.save(agent.critic_local1.state_dict(), 'checkpoint_critic1.pth')
            torch.save(agent.actor_local2.state_dict(), 'checkpoint_actor2.pth')
            torch.save(agent.critic_local2.state_dict(), 'checkpoint_critic2.pth')
    return scores, mean_scores





def main():
    # Load the environment Tennis
    env_name = ".\Tennis_Windows_x86_64\Tennis.exe"
    no_graphics = True
    env = UnityEnvironment(file_name = env_name, no_graphics = no_graphics)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode = True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print(f"Number of agents : {num_agents}")

    # size of each action
    action_size = brain.vector_action_space_size
    print(f"Size of each action : {action_size}")

    # examine the state space
    states = env_info.vector_observations
    _, state_size = states.shape
    print(f"There are {num_agents} agents. Each observes a state with length {state_size}")
    print("The state for the first agent looks like:", states[0])
    print("The state for the second agent looks like:", states[1])


    # create an agent
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random_seed  = 12345
    agent = Agent(state_size, action_size, num_agents, random_seed, device=device)
    start = time.time()
    scores, mean_scores = train_ddpg_v1(env, agent, n_episodes = 6000)
    print("Duration: {}".format(time.time() - start))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.plot(np.arange(1, len(mean_scores)+1), mean_scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    # # Take Random Actions in the Environment
    # for _ in range(1):
    #     env_info = env.reset(train_mode = False)[brain_name] # reset the environment
    #     states = env_info.vector_observations # get the current state (for each agent)
    #     scores = np.zeros(num_agents)
    #     while True:
    #         actions = np.random.randn(num_agents, action_size) # select an action for each agent
    #         actions = np.clip(actions, -1, 1)
    #         env_info = env.step(actions)[brain_name] # send all actions to the environment
    #         next_states = env_info.vector_observations # get next_state (for each agent)
    #         print(next_states)
    #         rewards = env_info.rewards # get reward (for each agent)
    #         dones = env_info.local_done # see if episode finished
    #         scores += rewards       # update the score for each agent
    #         states = next_states   # roll over states to the next time step
    #         if np.any(dones):
    #             break
    #     print("Total score (averaged over agents) for this episode: {}".format(np.mean(scores)))

    env.close()


if __name__ == "__main__":
    main()