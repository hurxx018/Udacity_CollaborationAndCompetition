from unityagents import UnityEnvironment


import numpy as np

import torch

from ddpg_agent import Agent

def main():

    # load version 2 (with 20 agents) of the environment
    env_name = "Tennis_Windows_x86_64\Tennis.exe" # add a Unity-Environment name.
    no_graphics = False
    env = UnityEnvironment(file_name = env_name, no_graphics = no_graphics)

    # Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the enviroment
    env_info = env.reset(train_mode = False)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print("Number of agents : ", num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print("Size of each action : ", action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print("There are {} agents. Each observes a state with length: {}".format(states.shape[0], state_size))


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random_seed = 12345 #10
    agent = Agent(state_size, action_size, num_agents, random_seed, device=device)
    actor_state_dict1 = torch.load("checkpoint_actor1.pth")
    agent.actor_local1.load_state_dict(actor_state_dict1)
    critic_state_dict1 = torch.load("checkpoint_critic1.pth")
    agent.critic_local1.load_state_dict(critic_state_dict1)
    actor_state_dict2 = torch.load("checkpoint_actor2.pth")
    agent.actor_local2.load_state_dict(actor_state_dict2)
    critic_state_dict2 = torch.load("checkpoint_critic2.pth")
    agent.critic_local2.load_state_dict(critic_state_dict2)



    # Take Random Actions in the Environment
    env_info = env.reset(train_mode=False)[brain_name]      # reset the environment
    states = env_info.vector_observations                  # get the current state (for each agent)

    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
        actions = agent.act(states, add_noise = False) # select an action (for each agent)
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished

        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


    # When finished, you can close the environment
    env.close()


if __name__ == "__main__":
    main()
