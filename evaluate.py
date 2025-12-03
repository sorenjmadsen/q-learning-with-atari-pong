import torch
from src.utils import make_testing_env
from src.network import QNetwork
from src.agent import InferenceQAgent
from pathlib import Path
from omegaconf import OmegaConf
import os

CONFIG_PATH = Path(__file__).with_name('config.yaml')
file_config = OmegaConf.load(CONFIG_PATH) if CONFIG_PATH.exists() else OmegaConf.create()
cli_config = OmegaConf.from_cli()
config = OmegaConf.merge(file_config, cli_config)

eval_config = config.get('evaluation', {})
weights_path = eval_config.get('weights_path', '')
device = eval_config.get('device', 'cpu')
environment_id = eval_config.get('environment_id', 'PongNoFrameskip-v4')

test_env = make_testing_env(environment_id, model_name=os.path.split(weights_path)[-1].split('.')[0])
observation = test_env.reset()

q_network = QNetwork(input_shape=observation.shape, action_size=test_env.action_space.n)
q_network.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device))

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