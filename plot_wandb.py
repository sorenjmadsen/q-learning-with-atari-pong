import wandb
import matplotlib.pyplot as plt
import os.path as op
runs = [
    # 'sorenjmadsen/DoubleDQN-ExperienceBuffer-PongNoFrameskip-v4/0tcw5lbw',
    # 'sorenjmadsen/DoubleDQN-PERBuffer-PongNoFrameskip-v4/2on90shc',
    'sorenjmadsen/DoubleDQN-PERBuffer-Pong-v4/v0kndv92',
    'sorenjmadsen/DoubleDQN-PERBuffer-Pong-v4/fy5nqko3'
    'sorenjmadsen/DoubleDQN-ExperienceBuffer-Pong-v4/d0tbmw8u'
    'sorenjmadsen/DoubleDQN-ExperienceBuffer-Pong-v4/760e5muq'

]

names = [
    'PERBuffer-Pong-v4 Default'
    'PERBuffer-Pong-v4 Adjusted'
    'ExperienceBuffer-Pong-v4 Default'
    'ExperienceBuffer-Pong-v4 Adjusted'
]

api = wandb.Api(timeout=120)

data = {}

for name, r in zip(names, runs):
    print(r)
    run = api.run(r)
    print(run)
    hist = run.history() 
    plt.plot(hist['local_q_mean'], label=name)

plt.xlabel('Train Step')
plt.ylabel('Smoothed Batch Q-Value')
plt.legend()
plt.title("Local Network Q-Value over Training Episodes")
plt.savefig(op.join('plots', 'figures', 'Q-Values.png'))
