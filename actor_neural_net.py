# ვიყენებ actor critic learning -ს 
# ეს კლასი არის ნეირონული ქსელის იმპლემენტაცია 
# რომელსაც იყენებს actor - კლასი
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor_Neural_Net(nn.Module):
    def __init__(self):
        super(Actor_Neural_Net, self).__init__()
        
        #ინფუთი არის state რომელიც არის 27 განზომილების ტენზორი
        self.input_to_hidden_layer1 = nn.Linear(27, 512)
        self.input_to_hidden_layer1.weight.data.normal_(0, 0.1)
        
        # მაქ პირველი hidden layer ნორმალირი განაწილებით ვარანდომებ weight-ებს    
        self.hidden_layer1_to_hidden_layer2 = nn.Linear(512, 400)
        self.hidden_layer1_to_hidden_layer2.weight.data.normal_(0, 0.1)
        
        # მეორე hidden layer
        self.hidden_layer2_to_hidden_layer3 = nn.Linear(400, 128)
        self.hidden_layer2_to_hidden_layer3.weight.data.normal_(0, 0.1)
        
        # output layer სადაც აუთფუთი არის 8 განზომილების ტენზორი (8 იმიტომ ტომ action არის = 8 ზომის)
        self.hidden_layer3_to_output_layer = nn.Linear(128, 8)
        self.hidden_layer3_to_output_layer.weight.data.normal_(0, 0.1)
    
    def forward(self,input_neural_net):
        
        #ვუშვებ ნეირონულ ქსელში ინფუთს
        input_neural_net=self.input_to_hidden_layer1(input_neural_net)
        input_neural_net=F.relu(input_neural_net)

        # პირველი hidden layer აქტივაციის ფუნქცია არის რელუ 
        input_neural_net=self.hidden_layer1_to_hidden_layer2(input_neural_net)
        input_neural_net=F.relu(input_neural_net)

        # მეორე hidden layer აქტივაციის ფუნქცია არის სიგმოიდი რადგან აქაც რელუ რომ მქონდეს 
        # ანუ [0,+inf] მაპინგი რომ მოვახვედრო მერე tanh რომ მოედება მარტო tanh(0,+inf)  -> (0,1) მაპინგი მოხდება
        # მე კიდე [-1,1] მინდა
        input_neural_net=self.hidden_layer2_to_hidden_layer3(input_neural_net)
        input_neural_net=torch.sigmoid(input_neural_net)

        #ბილოში მაქ tanh რადგან მინდა ვალიდური acion = [-1,1] მივიღო
        input_neural_net=self.hidden_layer3_to_output_layer(input_neural_net)
        input_neural_net = torch.tanh(input_neural_net)
        
        # ვაბრუნებ action-ს
        action = input_neural_net
        return action