# Q-Learning with Atari Pong

I developed a fascination for teaching computers to learn from environments. This is my foray into learning about the foundations of deep RL. For my experiments, I implemented a Double DQN from the original DeepMind paper (with some small tweaks).

## Quickstart

There's a `requirements.txt`, so just run `pip install -r requirements.txt` in whatever environment you are using. As a side note: I ran my repo on RunPod, and there was weird conflict with the previously installed version of `pyparsing` on the Pod. I managed to circumvent this with `pip install -r requirements.txt --ignore-installed pyparsing`.

### Training

To train, just run :
```python train.py```
to run with a replay buffer and the default hyperparameters in `config.yaml`.

### Evaluation

To evaluate the pretrained model weights just run:
```python evaluate.py```
or if you want to run with your own weights:
```python evaluate.py evaluation.weights_path=path/to/weights.pth```

When running evaluation, the testing environment will output a video into `recording/` for those of you that like to watch.

**You can also adjust the hyperparameters from the CLI.** It's as simple as a dot reference argument passed to the command. For example, you can adjust the batch size during training with:

```python train.py training.batch_size=64```

Check out the docs for OmegaConf for more information.

## Some Notes on Minor Tweaks
I was prototyping on my laptop for a while, so I adjusted the feature extraction portion of the network a little bit. The original paper had only 3 convolutions wither filters: 32, 64, 64. I adjusted to 32, 64, 128, 128 to make the linear layers a little bit smaller. This made prototyping and debugging a lot easier. 

Additionally, the Prioritized Experience Replay buffer uses multiplicative growth for annealing `beta` rather than linear steps. This made sense to me when I designed the class because I didn't need to keep track of a linear step. I might change this in future iterations.

## Things I Learned

1. Perhaps, the most unintuitive finding from my work is in regards to batch size. Larger batch sizes do not seem to improve learning. This [paper](https://openreview.net/pdf?id=wPqEvmwFEh) seems to suggest that Pong tends to favor smaller batches. 

2. Prioritized Experience Replay is hard to tune. Pong may be too simple of an environment to see a strong advantage from the implementation.

3. Debugging your own work is the best way to get intimately acquainted with the algorithms. I first embarked on this project 3 years ago, and I never finished (nothing ever converged). In the last few weeks, I've facepalmed so many times that I think I'm starting to iron out my forehead wrinkles. As an example, I leared to ALWAYS CHECK THE OBSERVATIONS YOU PASS TO YOUR MODEL. I found that my `ExperienceBuffer` and `PERBuffer` classes had mixed up the return order of `states, actions, rewards, next_states, dones`. This turned `next_states` into booleans. It turns out that it's pretty hard to determine the expected values of an observation state that falls out of the training distribution (/s). 

4. I wish I had the compute resources of DeepMind circa 2015. I can only imagine what they've got now with all those Ironwoods.

5. I am fascinated by the plethora of different reinforcement learning algorithms. So far, the off-policy algorithms make the most sense to me, but I am eager to implement PPO or A2C. Keep an eye out for updates.

## Setup

### Hardware

I did a ton of prototyping on my laptop (MacBook Air M2), but I decided that I wanted to also use it for other things. This inspired me to spin up a RunPod, but I'm stingy so I went for their cheapest option: the RTX 4000 Ada. This let me run 2-3 experiments at a time (RAM constrained).

### Gynamsium Environments

I noticed that a basic replay buffer seems to be the fastest to converge to an average score above 19.5, but PERBuffer never quite got all the way there. I tried both flavors of replay buffer on two variants of Pong: `Pong-v4` and `PongNoFrameskip-v4`. In total, `Pong-v4` seems to take longer for full convergence \(although it does start to win more frequently\). 

## Results

Overall, I'm happy that all the models eventually learned to win. I would have loved to see higher performance on `Pong-v4` with the similar architecture. Given the faster pace of the environment, it would make sense that each episode has fewer steps and therefore fewer training steps. However, doubling the maximum number of episodes did not help, and there's no clear upward trend after an average score of around 10. Despite the lack in performance, the videos show that the models exhibit some very interesting behavior. Both of the models trained on the two flavors of replay buffer seem to have found an optimal paddle location for winning. By placing the paddle a bit below halfway on the vertical axis, it seems that there's a good chance to win the game. Of course, when the ball goes elsewhere, the point is lost. Check them out in `recording/`! They made me crack up.

Here's some of the training curves that I saw:

![Average Score over Training Episodes](plots/figures/AverageScores-Default.png)

### Q-Value Inspection

During the training process, I decided to save the batch-average local and target network Q-values. This allowed me to check out how each of the Q-networks was learning. I decided to compare the localt network training Q-values 4 different training runs, and I found that the `Pong-v4` Q-functions seem to be overestimating. Additionally, this plot does seem to show the significant training time variation. Despite that piece, the estimated Q-values seem to be much larger for `Pong-v4` than `Pong-NoFrameskip-v4`.

![Average Batch Local Network Q-Value over Training Steps](plots/figures/Q-Values.png)

Given this overestimation, it seems to me that it's possible that the discount factor, `gamma`, is too high for the `Pong-v4` environment. In the case of `Pong-NoFrameskip-v4`, `gamma` set to `0.99` made sense given that a reward typically manifests every 60-120 steps. Therefore, if a reward in `Pong-v4` comes every 30-60 steps, then the discount factor must be reduced in order to prevent overestimation. In addition, a `batch_size` of 32 may be too large given that each batch may have a more diverse sampling of the environment. For these reason, I have started retraining the models on `Pong-v4` with a `gamma=0.97` and `batch_size=16`. 
