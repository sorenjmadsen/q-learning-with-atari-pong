import torch
from src.utils import make_testing_env
from src.network import QNetwork
from src.agent import InferenceQAgent

test_env = make_testing_env('PongNoFrameskip-v4')
observation = test_env.reset()

device = 'mps'
q_network = QNetwork(input_shape=observation.shape, action_size=test_env.action_space.n)
q_network.load_state_dict(torch.load('checkpoints/replay-checkpoint.pth', weights_only=True, map_location=device))

agent = InferenceQAgent(q_network=q_network, device=device)

done = False
rewards = []
while True:    
    #your agent goes here
    
    action = agent.play(observation, 0.01)
         
    observation, reward, done, trunc, info = test_env.step(action) 
    rewards.append(reward)
    # plt.imshow(observation) 
    # print(done)  
    if done or trunc: 
      break

print("Total Reward:", sum(rewards))
test_env.close()