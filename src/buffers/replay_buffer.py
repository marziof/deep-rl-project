# file to store experience for off-policy algos

import random
import numpy as np
import random
from collections import deque # double ended queue - uses append, pop (default right side), appendleft, popleft (left side)

#-----------------------
# Simple replay buffer implementation using deque
#-----------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
                            
    def store(self, state, action, reward, next_state, done):
        # store a single transition in the buffer - added as a tuple to the right-end of the deque (bottom of the stack)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # sample a random batch of transitions from the buffer
        # return as separate arrays for states, actions, rewards, next_states, and dones
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        # return the current size of the buffer
        return len(self.buffer)