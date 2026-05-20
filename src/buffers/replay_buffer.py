# file to store experience for off-policy algos

import numpy as np

class ReplayBuffer:
    """
    Vectorized numpy ring buffer for O(1) sampling.
    Pre-allocates arrays to avoid repeated allocations and deque O(n) indexing.
    """
    def __init__(self, capacity, state_dim=None, action_dim=None):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        # Lazy initialization: dimensions will be set on first store() call
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.initialized = False
        self.states = None
        self.actions = None
        self.rewards = None
        self.next_states = None
        self.dones = None
    
    def _initialize_arrays(self, state_dim, action_dim):
        """Initialize numpy arrays on first call."""
        if self.initialized:
            return
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.initialized = True
    
    def store(self, state, action, reward, next_state, done):
        """Store a single transition in O(1) time."""
        # Initialize on first store
        if not self.initialized:
            state_dim = np.array(state).shape[0] if np.isscalar(state) is False else 1
            action_dim = np.array(action).shape[0] if np.isscalar(action) is False else 1
            self._initialize_arrays(state_dim, action_dim)
        
        i = self.ptr
        self.states[i] = np.asarray(state, dtype=np.float32)
        self.actions[i] = np.asarray(action, dtype=np.float32)
        self.rewards[i] = np.asarray([[reward]], dtype=np.float32)
        self.next_states[i] = np.asarray(next_state, dtype=np.float32)
        self.dones[i] = np.asarray([[done]], dtype=np.float32)
        
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample a random batch in O(batch_size) time (not O(batch_size × buffer_size))."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx]
        )

    def __len__(self):
        """Return the current number of stored transitions."""
        return self.size
    
