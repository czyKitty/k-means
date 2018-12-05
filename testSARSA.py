import gym
import argparse
import random
from rl import *


def main():

    parser = argparse.ArgumentParser(description='Test the SARSA algorithm with OpenAI Gym.')
    parser.add_argument('-p', '--environment', type=str,   default='CartPole-v0', help='Specify an OpenAI environment to test (default: CartPole-v0)')
    parser.add_argument('-t', '--episodes',    type=int,   default=500,           help='Specify the number of episodes to test')
    parser.add_argument('-a', '--alpha',       type=float, default=0.1,           help='The step-size alpha (default: 0.1)')
    parser.add_argument('-e', '--epsilon',     type=float, default=0.1,           help='The exploration rate epsilon (default: 0.1)')
    parser.add_argument('-g', '--gamma',       type=float, default=0.9,           help='The discount factor gamma (default: 0.9)')
    parser.add_argument('-i', '--initQ',       type=float, default=0,             help='The initial Q-value')
    parser.add_argument('-n', '--numtilings',  type=int,   default=5,             help='The number of tilings to use')
    parser.add_argument('-s', '--numtiles',    type=int,   default=9,             help='Each tiling will divide the space into an NxN grid')
    parser.add_argument('-l', '--_lambda',    type=int,   default=0.9,             help='lambda for ET')
    
    args = parser.parse_args()
    env = gym.make(args.environment)
    numActions = env.action_space.n
    numDims = len(env.observation_space.high)
    
    ranges = []
    numTiles = []
    for i in range(numDims):
        ranges.append([env.observation_space.low[i], env.observation_space.high[i]])
        numTiles.append(args.numtiles)

    print("ranges and numTiles initialized!")
    print("ranges: "+str(ranges))
    print("numTiles: "+str(numTiles))
        
    featureGenerator = TileFeatures(ranges, numTiles, args.numtilings)

    print("featureGenerator initialized!")
    print("number of features: %d" % (featureGenerator.getNumFeatures()))

    numFeatures = featureGenerator.getNumFeatures()
    
    print("numFeatures %d, numActions %d" % (numFeatures, numActions))
    
    agent = LinearSarsaLearner(numFeatures, numActions, args.alpha, args.epsilon, args.gamma, args._lambda)

    print("Agent initialized!")

    
    for ep in range(args.episodes):

        totalR = 0
        
        activeFeatures = featureGenerator.getFeatures(list(env.reset()))
        print("activeFeatures: "+str(activeFeatures))        
        env.render()
        
        action = agent.epsilonGreedy(activeFeatures)
        print("action: "+str(action))
        
        state, reward, done, _ = env.step(action)
        env.render()

        print("state: "+str(state))
        
        while not done:
            newFeatures = featureGenerator.getFeatures(state)
            action = agent.learningStep(activeFeatures, action, reward, newFeatures)
            activeFeatures = newFeatures
            
            ret = env.step(action)
            state   = list(ret[0])
            totalR += ret[1]
            done    = ret[2]            
            env.render()
            
        if done:
            agent.terminalStep(activeFeatures, action, reward)
    
    
    env.close()
    
    return

main()
    
    
