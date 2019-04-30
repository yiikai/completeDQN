#!/usr/bin/env python
# coding: utf-8

import gym
import numpy as np
import collections 
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import random
from PriorityMemory import Memory


class DQNAgent:
    def __init__(self, state_size,action_size,supportDDQN = False,
                  supportPriority = True,ProiSize = 2000,Ratio = 20):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = collections.deque(maxlen=2000)
        self.gamma = 0.9    
        self.epsilon = 0.5
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.support_DDQN = supportDDQN
        self.support_priority = supportPriority
        self.ddqupdateRatio = Ratio
        self.model = self._build_model()
        if supportDDQN:
            self.target_model = self._build_model()
            self.target_model.set_weights(self.model.get_weights())
        if supportDDQN:
            self.priorBuffer = Memory(2000)
    
    def _build_model(self):
        model = Sequential()
        model.add(Dense(16, input_dim=self.state_size, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) == self.memory.maxlen:
            self.memory.popleft()
        self.memory.append((state, action, reward, next_state, done))
    
    def greedy_act(self,state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def act(self,state):
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self,batch_size):
        
        minibatch = random.sample(self.memory, batch_size)
        states = []
        targets = []
        
        for state, action, reward, next_state, done in minibatch:
            states.append(state[0])
            target_True = reward
            if not done:
                if not self.support_DDQN:
                    target_True = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
                else:
                    target_True = (reward + self.gamma * np.amax(self.target_model.predict(next_state)[0]))
            traget_Current = self.model.predict(state)
            traget_Current[0][action] = target_True
            targets.append(traget_Current[0])
            
        history = self.model.fit(np.array(states), np.array(targets),batch_size = batch_size, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        if self.epsilon > self.epsilon_min:
            self.epsilon -= (0.5 - self.epsilon_min)/10000
            #self.epsilon *= self.epsilon_decay
        return loss
        
    def prior_replay(self,minibatch,ISWeights):
        
        states = []
        targets = []
        for state, action, reward, next_state in minibatch:
            states.append(state[0])
            target_True = reward
            if not done:
                if not self.support_DDQN:
                    target_True = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
                else:
                    target_True = (reward + self.gamma * np.amax(self.target_model.predict(next_state)[0]))
            traget_Current = ISWeights*self.model.predict(state)
            traget_Current[0][action] = ISWeights*target_True
            targets.append(traget_Current[0])
            
        history = self.model.fit(np.array(states), np.array(targets),batch_size = batch_size, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        if self.epsilon > self.epsilon_min:
            self.epsilon -= (0.5 - self.epsilon_min)/10000
            #self.epsilon *= self.epsilon_decay
        update_prior = []
        for state, action, reward, next_state in minibatch:
            td_error = np.abs(self.model.predict(state) - reward + self.gamma * np.amax(self.target_model.predict(next_state)[0]))
            update_prior.append(td_error)
        return loss,update_prior
    
        
            
    def training(self,reward,state,next_state,done):
        if not done:
            if not self.support_DDQN:
                target_True = reward + self.gamma * (np.amax(self.model.predict(next_state)[0]))
            else:
                target_True = reward + self.gamma * (np.amax(self.target_model.predict(next_state)[0]))
        else:
            target_True = reward
        target_Current = self.model.predict(state)
        target_Current[0][action] = target_True
        self.model.fit(state, target_Current, epochs=1, verbose=0)
        
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


env = gym.make('CartPole-v1')
actsize = env.action_space.n
stateSize = env.observation_space.shape[0]
agent = DQNAgent(stateSize,actsize,supportDDQN = True,)


episodes = 1000
maxtimesteps = 200
replay_batchsize = 32

count =0
for i in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, stateSize])
    for step in range(maxtimesteps):
        #env.render()
        action = agent.greedy_act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, stateSize])
        if agent.support_priority:
            transition = np.hstack((state, np.array([action, reward]).reshape(1,-1), next_state))
            agent.priorBuffer.store(transition)
            #agent.priorBuffer.store((state, action, reward, next_state, done))
        else:
            agent.remember(state, action, reward, next_state, done)
        #agent.training(reward,state,next_state,done)
        state = next_state
        if done:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}"
                      .format(i, episodes, step))
                break
        if agent.support_priority:
            count += 1
            if (count != 0) and (count % agent.ddqupdateRatio == 0):
                tree_idx, batch_memory, ISWeights = agent.priorBuffer.sample(replay_batchsize)
                _,update = agent.prior_replay(batch_memory, ISWeights)
                agent.priorBuffer.batch_update(tree_idx,update)
        else:
            if len(agent.memory) > replay_batchsize:
                count += 1
                agent.replay(replay_batchsize)
                if count % agent.ddqupdateRatio == 0:
                    agent.target_model.set_weights(agent.model.get_weights()) 

     



agent.save('cartpole_weights200steps')


for epsiod in range(100):
    state = env.reset()
    for t in range(500):
        state = np.reshape(state, [1, stateSize])
        env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        if done:
            print('Failed at {}'.format(t))
            break;
        state = next_state
env.close()
