import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import pickle

import Actor
import Critic
import memory_states

#გააკეთე environment
env = gym.make('Ant-v4').unwrapped

# დააინიციალიზე actor,criric და memory
actor=Actor.Actor()
critic=Critic.Critic()
Memory = memory_states.Memory()

# გაუშვი სიმულაცია 1000 ჯერ
for simulation in range(1000):
    print(simulation)

    # ყოველ 100-ის ჯერადზე შეინახე მოდელები actor და critic
    if simulation % 100 == 0:
        actor.load_weights()
        save_path = 'actor_object.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(actor, f)
        
        save_path = 'critic_object.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(critic, f)


    reward_single=0

    #დააინიციალიზე environment
    observation,info = env.reset()

    #გაუშვი 500 timestep 
    for timestep in range(500):
        
        # აიღე action - აქტოროდან
        action=actor.select_action(observation).detach().numpy()
        
        # შეასრულე action
        observation_next, reward, terminated, truncated, info = env.step(action)
        Memory.add(observation,action,reward,observation_next)
        
        # თუ საკმარისი სამფლი დაგროვდა დაააბდეითე ctitic და actor
        if Memory.counter>12000:

            #აიღე სამფლი მემორიდან
            states , actions , rewards , next_states = Memory.sample(100) 
            actor_action = actor.select_action(states)
            actor_action_ = actor.select_action_from_target_network(next_states)
            critic.train(states, actions, rewards, next_states, actor_action_)


            #გაუშვი training
            Q_c_to_a_loss = critic.ret(states, actor_action)
            actor.train(Q_c_to_a_loss)
                
            # დაააბდეითე target network-ები main network-ების მიხედვით
            actor.update()
            critic.update()

        # თუ სიმულაცია შშეწყდა შეაჩერე environment
        if terminated or truncated:
            break

        # bservation = observation_next, reward_single += reward
        observation = observation_next
        reward_single += reward

    #დაბეჭდე reward
    print(reward_single)

#დაანთავრე სიმულაცია
env.close()