# რადგან state space და policy state - ორივე არადისკრეტულია ვიყენებ actor critic learning-ს
# ეს კლასი გამოიყენება critic-ის მიერ
#  ნეირონული ქსელით რომ დააგენერიროს observation და action - ის მიხედვით Q_value 

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Critic_Neural_Net(nn.Module):
    def __init__(self):
        super(Critic_Neural_Net, self).__init__()
        
        # ინფუთის ზომა არი 27(state)+8(actor output) = 35
        self.in_to_y1 = nn.Linear(35, 256)
        self.in_to_y1.weight.data.normal_(0, 0.1)
        
        # მაქვს hidden layer - ები 
        # 35->256->32->1
        # წონები დაინიციალიზებულია Random ნორმალური განაწილების მიხედვით
        
        self.y1_to_y2 = nn.Linear(256, 32)
        self.y1_to_y2.weight.data.normal_(0, 0.1)
        
        self.out = nn.Linear(32, 1)
        self.out.weight.data.normal_(0, 0.1)
        
        # აუთფუთის ზომა არის 1 ცალი Q-value
    
    def forward(self, s, a):

        # ვაინიცირებ weight-ებს ნორმალური განაწილებით რანდომად
        input_neural_net = torch.cat((s, a), dim=1)
        input_neural_net = self.in_to_y1(input_neural_net)
        input_neural_net = F.relu(input_neural_net)
        input_neural_net = self.y1_to_y2(input_neural_net)
        input_neural_net = F.relu(input_neural_net)
        
        return self.out(input_neural_net)
    
        # აქ პირიქით ვაკეთებ ვიდრე actor-ის ნეირონულ ქსელში 
        # აქ არ ვდებ აქტივაციის ფუნქციას ბოლოში პირდაპირ წრფივ ლეიერში ვუშვებ და მერე აღარაფერს ვდებ რადგან critic -ს აუთფუთი -inf +inf -ში უნდა იყოს და 
        # რელუ, ტანგესი ან სიგმოიდი არევს მაგას