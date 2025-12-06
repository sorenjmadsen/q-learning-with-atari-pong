import wandb
import matplotlib.pyplot as plt
import os.path as op
runs = [
    'sorenjmadsen/DoubleDQN-ExperienceBuffer-PongNoFrameskip-v4/0tcw5lbw',
    'sorenjmadsen/DoubleDQN-PERBuffer-PongNoFrameskip-v4/2on90shc',
    'sorenjmadsen/DoubleDQN-PERBuffer-Pong-v4/v0kndv92',
    'sorenjmadsen/DoubleDQN-ExperienceBuffer-Pong-v4/d0tbmw8u'
]

api = wandb.Api()

data = {}

for r in runs:
    print(r)
    run = api.run(r)
    name = r.split('/')[1]
    hist =run.history() 
    plt.plot(hist['_step'], hist['loss'], label=name)

plt.xlabel('Train Step')
plt.ylabel('Loss')
plt.legend()
plt.title("Local Network Loss over Training Episodes")
plt.savefig(op.join('plots', 'figures', 'Loss.png'))
