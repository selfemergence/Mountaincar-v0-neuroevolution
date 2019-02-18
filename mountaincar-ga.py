import time, math, random, bisect, copy
import gym
import numpy as np
import os

from copy import deepcopy

#np.random.seed(123456789)
#random.seed(1)

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def tanh(x, derivative=False):
    if (derivative == True):
        return (1 - (x ** 2))
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2

def relu(x, derivative=False):
    return x * (x > 0)

def linear(x, derivative=False):
    return x

def roulette_selection(pop):
    """
    Assumes the population already sorted
    use the roulet-wheel to choose
    one individual each time to form the breeding pool"""
#    pop.sort(reverse=True)
    total = sum([c.fitness for c in pop])
    pick = np.random.uniform(0, total)
    current = 0
    for ind in pop:
        current += ind.fitness
        if current > pick:
            return ind
    return pop[0]

def tournament_selection(pop, toursize=2):
    """
    Assumes the population already sorted
    use the tournament-selection to choose
    one individual each time to form the breeding pool"""
    samples = np.random.choice(pop, toursize, replace=False)
    return max(samples)


###################################################################################
### Define some basic objects ###
###################################################################################
class NeuralNet:
    def __init__(self, Weights):
        """ Weights is a tuple of numpy matrices, one for each network inbetween
        neurons. The matrices are mxn, where m = # neurons at the exit and
        n = # neurons at the entrance to the network """
        self.NLayers = len(Weights)
        self.Weights = Weights
        
        self.fixedWeights = deepcopy(self.Weights)
        
    def think(self, inputs): # inputs must be the same length as Weights[0], returns tuple of length Weights[end]
        self.post_synapses = []
        self.pre_synapses = []
        
        inputs = np.array(inputs, ndmin=2).T
        
        for i in range(self.NLayers):
            self.pre_synapses.append(inputs)
            inputs = np.dot(self.Weights[i], inputs)
            
            if i == self.NLayers-1:
                inputs = sigmoid(inputs)
            else:
                inputs = sigmoid(inputs)
            self.post_synapses.append(inputs)

        return inputs

class Agent : 
    def __init__(self, env, w = [], b = []):  
        
        # weights and biases for the actor brain
        if len(w) > 0:
            weights = w
        else:
            weights = [np.random.randn(1, env.observation_space.shape[0]),
#                       np.random.randn(10, 10),
                       np.random.randn(env.action_space.n,1)]

        self.brain = NeuralNet(weights)
        
        
        self.fitness = 0.0
        self.env = env
        
    def reproduce(self):
        child_weights = deepcopy(self.brain.fixedWeights)
        return Agent(self.env, child_weights)#, child_biases)
    
    def convex_mate(self, other):
        new_weights = deepcopy(self.brain.fixedWeights)
        for i in range(len(new_weights)):
            for j in range(len(new_weights[i])):
                for k in range(len(new_weights[i][j])):
                    c_w = np.random.uniform()
                    new_weights[i][j][k] = c_w*self.brain.fixedWeights[i][j][k] + (1-c_w)*other.brain.fixedWeights[i][j][k]
                    
        #mutation
        for weights in new_weights:
            for weight in  weights:
                for w in weight:
                    if np.random.random() < 0.05:
                        w = w + (np.random.random() - 0.5) * 0.1
                        
        return Agent(self.env, new_weights)
    
        
    def __lt__(self, other):
        return self.fitness < other.fitness
    
    def __le__(self, other):
        return self.fitness <= other.fitness
    
    def fitness_evaluation(self, epochs, steps):
        score = []
        for epoch in range(epochs):
            observation = self.env.reset()
            totalReward = 0
            st = 0
            while st < steps:
                st += 1
                output = self.brain.think(observation)
                action = int(np.argmax(output))
                
                observation, reward, done, info = self.env.step(action)
                
                if (st >= 199):
                    #print ("Failed. Time out")
                    done = True
                    reward -= 20            
                    
                if done and st < 199:
                    print ("Sucess at epoch ", epoch)
                    reward += 20
                
                totalReward += reward
                if done:
                    break
                
            score.append(totalReward)
            
            if np.average(score) >= self.env.spec.reward_threshold:
                print("Solved at epoch ", epoch)
                
        self.fitness = sum(score)/len(score)

    
    def replay(self, steps):
        observation = self.env.reset()
        totalReward = 0
        for step in range(steps):
            self.env.render()
            time.sleep( 0.0625)
            output = self.brain.think(observation)
            action = int(np.argmax(output))
            
            observation, reward, done, info = self.env.step(action)
            totalReward += reward
            
            if done:
                break
            
        self.new_fitness = totalReward
    
class Evolution:
    def __init__(self, env, pop_size=20, gens=51, elitism=0, steps=1000, epochs=20):
        self.agents = [Agent(env) for _ in range(pop_size)]
        self.pop_size = pop_size
        self.gens = gens
        self.steps = steps
        self.epochs = epochs
        self.elitism = elitism
        
        self.env = env
        
    def evolve(self):
        for gen in range(self.gens):
            print('>Gen ', gen)
            
            for agent in self.agents:
                # fitness evaluation
                agent.fitness_evaluation(self.epochs, self.steps)
                
            # sort population
            self.agents.sort(reverse=True)
            
            print("\tMIN ", self.get_min().fitness, " | AVG ", self.get_avg(), " | MAX ", self.get_best().fitness)
            best_agents.append(self.get_best())
            best_fitness.append(self.get_best().fitness)
            avg_fitness.append(self.get_avg())
            worst_fitness.append(self.get_min().fitness)
            
            print("\nCreating New Population")
             #new population
            new_pop = []
            
            elites = self.agents[:(int(1/10*len(self.agents)))]
            for elite in elites:
                new_pop.append(elite.reproduce())
            
            while len(new_pop) < self.pop_size:
                # selection for crossover
                parent1 = tournament_selection(self.agents)
                parent2 = tournament_selection(self.agents)
                    
                # do crossover to produce one child
                child = parent1.convex_mate(parent2)
                    
                # mutatethe child
#                child.mutate_after_crossover()
                    
                new_pop.append(child)
            
            # copy new pop
            self.agents = new_pop[:]
    
    def get_best(self):
#        return max([agent.fitness for agent in self.agents])
#        return self.agents[0]
        return max(self.agents)
    
    def get_min(self):
#        return min([agent.fitness for agent in self.agents])
#        return self.agents[-1]
        return min(self.agents)
    
    def get_avg(self):
        return np.average([agent.fitness for agent in self.agents], axis=0)
    
def recordBestBots(bestAgents, env, max_steps):  
    print("\n Recording Best Bots ")
    print("---------------------")
#    env.monitor.start('Artificial Intelligence/'+GAME, force=True)
    observation = env.reset()
    for i in range(len(bestAgents)):
        totalReward = 0
        for step in range(max_steps):
            env.render()
            output = bestAgents[i].brain.think(observation)
            action = np.argmax(output)
            
            new_obs, reward, done, info = env.step(action)
            if (step >= 199):
                #print ("Failed. Time out")
                done = True
                reward -= 20            
            observation = new_obs
            if done and step < 199:
                print ("Sucess!")
                reward += 20
                
            totalReward += reward
            
            if done:
                observation = env.reset()
                break
        print("Generation %3d | Expected Fitness of %4d | Actual Fitness = %4d" % \
              (i+1, bestAgents[i].fitness, totalReward))
        
    env.close()
   
    
best_agents = []
best_fitness = []
avg_fitness = []
worst_fitness = []

env = gym.make('MountainCar-v0')
np.random.seed(0)
env.seed(0)

pop_size = 100
gens = 101
elitism=0
steps = 1000
epochs = 20

evolution = Evolution(env, pop_size, gens, elitism, steps)

evolution.evolve()

np.savetxt('results/car-fitness.txt', list(zip(best_fitness, avg_fitness, worst_fitness)), \
               fmt='%.18g', delimiter='\t', header="Fitness: Best, Avg, Worst")

import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(best_fitness, 'r-')
plt.plot(avg_fitness, 'g:')
plt.plot(worst_fitness, 'b--')
plt.xlabel("Generation")
#plt.xlim(xmin=0)
#plt.ylim(ymin=0)
plt.ylabel("Fitness")
plt.legend(['Best', 'Avg', 'Worst'], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0., ncol=3, mode="expand")
plt.savefig('results/car.png')

    
print ("End of simulation")

#save the best
import pickle
pickle.dump(best_agents, open('results/car.p', 'wb'))


print("Replay the best ")

recordBestBots(best_agents, env, steps)

# Test
best = max(best_agents)
for i_episode in range(100):
    print(">Episode ", i_episode)
    observation = env.reset()
    totalReward = 0
    for step in range(1000):
        env.render()
        output = best.brain.think(observation)
        action = np.argmax(output)
        new_obs, reward, done, info = env.step(action)
        if (step >= 199):
            #print ("Failed. Time out")
            done = True
            reward -= 20            
                    
        if done and step < 199:
            print ("Sucess!")
            reward += 20
                
        totalReward += reward
        
        observation = new_obs
        if done:
            print("\tDone in ", step)
            print("\tTotal reward ", totalReward)
            print("\tEpisode finished after {} timesteps".format(step+1))
            break
env.close()

        
        
        

