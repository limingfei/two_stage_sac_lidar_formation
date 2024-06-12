import numpy as np
from cpprb import PrioritizedReplayBuffer

buffer_size = 256

prb = PrioritizedReplayBuffer(buffer_size,
                              {"obs": {"shape": (4,4)},
                               "act": {"shape": 3},
                               "rew": {},
                               "next_obs": {"shape": (4,4)},
                               "done": {}},
                              alpha=0.5,
                              next_of='obs')

for i in range(1000):
    prb.add(obs=np.zeros((4,4)),
            act=np.ones(3),
            rew=0.5,
            next_obs=np.zeros((4,4)),
            done=0)

batch_size = 32
s = prb.sample(batch_size,beta=0.5)

for k in range(1000):
    indexes = s["indexes"]
    weights = s["weights"]
    td_errors = np.random.normal(0,10,size=(32,))
    #  train
    #  ...
    print(weights)
    # print(indexes)
    new_priorities = np.ones_like(td_errors)
    # print(td_errors)
    # print(new_priorities)
    prb.update_priorities(indexes,td_errors)