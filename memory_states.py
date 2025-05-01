# ეს კლასი არის მემორის იმპლემენტაცია
# რომელიც შეინახავს (state, action, reward, next_state) - ს
# state 27 + action 8 + reward 1 + next state 27 = 63 განზომილება
import numpy as np

class Memory():

    # ინიციალიზება ცარიელი მემორის 
    def __init__(self):
        self.capacity = 20000
        self.mem = np.zeros((20000, 63))
        self.counter = 0

    # შეინახე მემორიში
    # წადი წრეზე ანუ 20001 ელემენტი პირველს გადააწერე
    def add(self, state, action, reward, next_state):
        transition = np.hstack((state, action, reward, next_state)) 
        index = self.counter % self.capacity
        self.mem[index, :] = transition  
        self.counter += 1

    # დააბრუნე სამფლი 
    # n ცალი (state, action, reward, next_state)
    def sample(self, n):
        sample_index = np.random.choice(self.capacity, n)
        mem = self.mem[sample_index, :]
        return mem[:, :27],mem[:, 27: 27 + 8],mem[:, -27 - 1: -27],mem[:, -27:]