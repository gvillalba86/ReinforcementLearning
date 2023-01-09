import gym
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
import imageio
from pathlib import Path
from tqdm.notebook import tqdm


def play_episode(env:gym.Env, agent, save:bool = False) -> tuple[float, int]:
    '''Plays a random episode and returns the total reward and number of steps taken'''

    obs = env.reset()[0]
    total_reward, done = 0, False
    frames = []
    
    def _label_with_text(frame):
        '''
        frame: gym environment frame
        '''
        im = Image.fromarray(frame)
        im = im.resize((im.size[0]*2,im.size[1]*2))
        drawer = ImageDraw.Draw(im)
        drawer.text((1, 1), "UOC Aprendizaje Por Refuerzo. Gerson Villalba Arana", 
                    fill=(255, 255, 255, 128))
        return im
    
    for step in range(1, 1000):
        # Get action from agent
        action = agent.get_action(obs)
        if save:
            frame = env.render()
            frames.append(_label_with_text(frame))
        # Take step in environment
        new_obs, reward, done, info, _ = env.step(action)
        # Update state and total reward
        obs = new_obs
        total_reward += reward
        if done:
            break
    
    if save:
        filename = 'lunar_lander_' + agent.__class__.__name__ + '.gif'
        imageio.mimwrite(Path('videos') / filename, frames, fps=60)
        
    return total_reward, step


def play_n_episodes(env:gym.Env, agent, n_episodes:int) -> tuple[np.ndarray, np.ndarray]:
    '''Plays n_episodes random episodes and returns the total reward and number of steps taken'''
    total_rewards = []
    n_steps = []
    for i in tqdm(range(n_episodes)):
        total_reward, steps = play_episode(env, agent, save=False)
        total_rewards.append(total_reward)
        n_steps.append(steps)
    return np.array(total_rewards), np.array(n_steps)

