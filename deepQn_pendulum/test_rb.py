import numpy as np
from dqn_agent import ReplayBuffer
import torch
import random
"""
class somebuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = [None]*buffer_size
        self.idx = 0

    def insert(self,someVal):
        self.buffer[self.idx % self.buffer_size] = someVal
        self.idx += 1

if __name__ == "__main__":

    buff_obj = ReplayBuffer(25)
    my_numbers = np.arange(0,250)

    for i in range(len(my_numbers)):
        buff_obj.insert_sample(my_numbers[i])
        print(buff_obj.idx % buff_obj.buffer_size)
        print(buff_obj.buffer)
"""
if __name__ == "__main__":
    my_tensor = torch.tensor([[0]], dtype = torch.long)
    print(my_tensor.size())
    for i in range(10):
        my_tensor[0][0]  = random.randrange(2)#torch.tensor([[random.randrange(2)]], dtype=torch.long)
        print(my_tensor)