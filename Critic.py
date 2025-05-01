# ვიყენებ actor critic learning -ს 
# ეს კლასი არის critic-ის იმპლემენტაცია 

import Critic_Neural_Net  
import gymnasium as gym 
import torch 
import torch.nn as nn  
import torch.nn.functional as F 
import numpy as np  

discount =  0.9
learning_rate = 0.0005

class Critic():

    # შექმენით critic ინიცირება ორი ნერვული ქსელით: ერთი evaluate და ერთი target
    # როგორც actor - სთვის გავაკეთე
    # ამის იმპლემენტაცია Critic_Neural_Net - კლასში მაქვს

    # დააყენეთ ოპტიმიზატორი შეფასების ქსელის ტრენინგისთვის
    #loss function = Mean Squared Error Loss    

    def __init__(self):
        self.main_network = Critic_Neural_Net.Critic_Neural_Net()
        self.copy_network = Critic_Neural_Net.Critic_Neural_Net()
        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=learning_rate)
        self.lossfun=nn.MSELoss()


    # გამოთვალეთ კრიტიკოსის შეფასების ნეგატივი q value (სასწავლო მიზნებისთვის)
    
    # ამის მაქსიმუმი ანუ -1 * ამის მინიმუმი მინდა gradient descent სთვის
    def ret(self, state, action):
        return -self.main_network(torch.FloatTensor(state), action).mean()


    # ტრეინინგის ფორმატი არის შემდეგი
    # Q(state) = Reward(state,action) + discount * Q(next_state)
    # ფუნქციას გადაეცემა  state, action, reward, state_next, action_next
    # და ცდილობს 
    # reward + დისკონტირებული target - ის q value შემდეგი სთეითისთვის მიაახლოვოს პირდაპირ აღებულ q value -ს 

    def train(self, state, action, reward, state_next, action_next):
        
        Q1 = self.main_network(torch.FloatTensor(state),torch.FloatTensor(action))

        Q_next = self.copy_network(torch.FloatTensor(state_next), action_next).detach()
        Q2 = torch.FloatTensor(reward) + discount * Q_next

        loss = self.lossfun(Q1, Q2)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # როგორც actor - ში მქონდა ისეთივე იმპლემენტაცია soft update - თვის 
    # ოღონდ critic - ის ნეირონული ქსელისთვის  

    def update(self):
        for target_param, param in zip(self.copy_network.parameters(), self.main_network.parameters()):
            target_param.data.copy_(target_param.data * 0.9995 + param.data * 0.0005)


        

