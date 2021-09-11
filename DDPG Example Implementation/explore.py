from collections import deque
import pandas as pd
import numpy as np
import gym
import sys

from buffer import ReplayBuffer
from agent import TD3

class Runner:
    """Carries out the environment steps and adds experiences to memory"""
    
    def __init__(self, algo, n_episodes=100, batch_size=32, gamma=0.99, tau=0.005, noise=0.2, noise_clip=0.5, explore_noise=0.1,\
            policy_frequency=2, sizes=None):
                
        self.algo = algo
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.noise = noise
        self.noise_clip = noise_clip
        self.explore_noise = explore_noise
        self.policy_frequency = policy_frequency
        self.replay_buffer = ReplayBuffer(algo)
        
        self.agent = TD3(self, state_dim=sizes[0], action_dim=sizes[1], max_action=sizes[2], seed=sizes[3])

    def evaluate_policy(self, TestEnv, eval_episodes=1):
        """
        run several episodes using the best agent policy
            
            Args:
                policy (agent): agent to evaluate
                env (env): gym environment
                eval_episodes (int): how many test episodes to run
                render (bool): show training
            
            Returns:
                avg_reward (float): average reward over the number of evaluations    
        """
        
        avg_reward = 0.
        for i in range(eval_episodes):
            obs = TestEnv.reset()
            done = False
            while not done:
                action = self.agent.select_action(np.array(obs), noise=0)
                obs, reward, done, _ = TestEnv.step(action)
                avg_reward += reward
                if action <= -0.05:
                    self.algo.Log("Action {}.".format(action))
                
        #self.algo.Log("Eval {}.".format(round(avg_reward,2)))

        avg_reward /= eval_episodes
        
        return avg_reward
    
    def observe(self, TrainEnv, observation_steps):
        """
        run episodes while taking random actions and filling replay_buffer
        
            Args:
                env (env): gym environment
                replay_buffer(ReplayBuffer): buffer to store experience replay
                observation_steps (int): how many steps to observe for    
        """
        
        time_steps = 0
        obs = TrainEnv.reset()
        done = False
        
        while time_steps < observation_steps:
            action = TrainEnv.action_space.sample()
            new_obs, reward, done, _ = TrainEnv.step(action)
    
            self.replay_buffer.add((obs, new_obs, action, reward, done))
    
            obs = new_obs
            time_steps += 1
    
            if done:
                obs = TrainEnv.reset()
                done = False
    
            #self.algo.Log("Populating Buffer {}/{}.".format(time_steps, observation_steps))
            #sys.stdout.flush()
        
    def train(self, TrainEnv, TestEnv):
        """
        Train the agent for exploration steps    
            Args:
                agent (Agent): agent to use
                env (environment): gym environment
                exploration (int): how many training steps to run    
        """
        
        scores = []
        scores_avg = []
        scores_window = deque(maxlen=25)
        
        eval_reward_best = -1000
        
        self.algo.Debug("{} | Training..".format(self.algo.Time))
        
        eval_reward = self.evaluate_policy(TestEnv, int(TestEnv.MaxCount))

        if eval_reward > eval_reward_best:
            eval_reward_best = eval_reward
            self.algo.Debug("Last Model Tested |"+str(eval_reward_best))
            self.agent.save("best_avg")
        
        for i_episode in range(1, self.n_episodes+1):
            
            obs = TrainEnv.reset()
            score = 0
            done = False
            episode_timesteps = 0
            
            while not done:
                
                action = self.agent.select_action(np.array(obs), noise=self.explore_noise)
                new_obs, reward, done, _ = TrainEnv.step(action)
                self.replay_buffer.add((obs, new_obs, action, reward, done))
                obs = new_obs
                score += reward
                
                episode_timesteps += 1
                
                scores_window.append(score)
                scores.append(score)
                scores_avg.append(np.mean(scores_window))
                
                self.agent.train(self.replay_buffer, episode_timesteps, self.batch_size, self.gamma, self.tau, self.noise, self.noise_clip, self.policy_frequency)
                    
            # how often do we want to eval model?
            if i_episode % 1 == 0:
                # get eval over all symbols and save best avg
                eval_reward = self.evaluate_policy(TestEnv, int(TestEnv.MaxCount))
                
                if eval_reward > eval_reward_best:
                    eval_reward_best = eval_reward
                    self.algo.Debug(str(i_episode)+"| Best Model! |"+str(round(eval_reward_best,3)))
                    self.agent.save("best_avg")
                    #self.algo.Log("{} {} {} {} {}".format(episode_timesteps, i_episode, score, eval_reward))