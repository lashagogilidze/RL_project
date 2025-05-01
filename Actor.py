# ვიყენებ actor critic learning -ს 
# ეს კლასი არის actor-ის იმპლემენტაცია 

# როგორც აქტორი რომელიც აგენერირებს action-ს და სწავლობს მის დაგენერირებას

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

import actor_neural_net
import Critic_Neural_Net

learning_rate = 0.0005

class Actor():

    # ინიციალიზაციის დროს ვქმნი 2 ნეირონულ ქსელს რომელიც Actor_Neural_Net კლასით გავაკეთე
    # პირველი ნეირონული ქსელი არის იმისთვის რომ observation დამაპოს action-ზე
    # მეორე ნეირონული ქსელი არის target სწავლებისთვის 
    # იგივე პირველი ნეირონული ქსელია მაგრამ მეტად სტაბილურად განახლებადი
    # θ1 <- τ⋅ θ2 + (1−τ)⋅ θ1 # ესაა დამოკიდებულება ორ ნეირონულ ქსელს შორის 
    # ანუ soft update მექანიზმი რომელიც ამ ორ ნეირონულ ქსელს აკავშირებს
    # (soft update) - ის იმპლემენტაცია მერე მექნება

    def __init__(self):
        self.main_network = actor_neural_net.Actor_Neural_Net()
        self.target_network = actor_neural_net.Actor_Neural_Net()
        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=learning_rate)

    # ეს ფუნქცია აბრუნებს action-ს observation -ის მიხედვით 
    # ანუ ნეირონულ ქსელში ინფუთად observation - ის შედეგს

    def select_action(self, state):
        return self.main_network(torch.FloatTensor(state))

    # ეს ფუნქცია აბრუნებს action-ს observation -ის მიხედვით 
    # ანუ ნეირონულ ქსელში ინფუთად observation - ის შედეგს
    # ოღონდ ვიყენებ ტარგეტ network-ს

    def select_action_from_target_network(self, state):
        return self.target_network(torch.FloatTensor(state)).detach()

    # ლოსის მიხედვით update-ს უკეთებს ნეირონულ ქსელს
    # loss - კრიტიკისგან იღებს

    def train(self, loss_from_criric):
        loss = loss_from_criric
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # ზემოთ ნახსენები soft update -ს იმპლემენტაცია

    def update(self):
        for target_param, param in zip(self.target_network.parameters(), self.main_network.parameters()):
            target_param.data.copy_(target_param.data * 0.9995 + param.data * 0.0005)

    #ინახავს მოდელის weight-ებს

    def load_weights(self):
        save_path = 'actor_neural_net_weights' 
        torch.save(self.main_network.state_dict(), 'actor_neural_net_weights.pt')
        
        print(f"Model weights saved to {save_path}")

