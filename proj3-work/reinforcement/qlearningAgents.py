# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from featureExtractors import *
from learningAgents import *

import random
import util
import math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.qValueList = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.qValueList[(state, action)]



    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        valueDict= util.Counter()
        #Check if we are in a terminal state
        if len(legalActions) == 0:
          return 0.0
        #Add a key of action for every legal action in the counter dictionary 
        for action in legalActions:
        #Assign the Q value for the key (action)
          valueDict[action] = self.getQValue(state, action)
        #return the maximum (the Value)
        return valueDict[valueDict.argMax()] 

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        optimalAction = None
        maxVal = float('-inf')
        legalActions = self.getLegalActions(state)
        for action in legalActions:
          qS = self.qValueList[(state, action)]
          if maxVal < qS:
            maxVal = qS
            optimalAction = action
        return optimalAction

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """


        "*** YOUR CODE HERE ***"
        # Pick Action
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:                        # Return None if no actions available
          return None
        else:
          if util.flipCoin(self.epsilon):                 # Occurs with probability epsilon
            return random.choice(legalActions)            # Choose an action randomly with uniform probability 
          else:
            return self.computeActionFromQValues(state)   # Choose action based on max Q value

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        qS = self.getQValue(state, action)
        qSPrime = self.getValue(nextState)
        alpha = self.alpha
        gamma = self.discount
        #If the agent did not make it to the next state
        if not nextState:
            self.qValueList[(state, action)] = (1-alpha) * qS + alpha * reward
        else:
            self.qValueList[(state, action)] = (1-alpha) * qS + alpha * (reward + gamma * qSPrime)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        totalValue = 0 # inital totalValue of QValue
        list_of_features = self.featExtractor.getFeatures(state, action) # list of the features given
        for i in list_of_features:
            if self.weights[i] == 0 or list_of_features[i] == 0: # if either value is 0 no reason to calculate
                continue
            totalValue += self.weights[i] * list_of_features[i] # Q-value is the sum of weights * its corresponding feature
        return totalValue # finally return the total Qvalue for the given state, action

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        diff = (reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state, action) # calculate the diffence with given formula from the question
        list_of_features = self.featExtractor.getFeatures(state, action) # get the list of features which is using the Counter class, similar to a dictionary
        for i in list_of_features: # iterate through the list of features
            self.weights[i] = self.weights[i] + self.alpha * diff * list_of_features[i] # update the weights with the fomula given in q10

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)


        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            for i in self.getWeights():
                print("I: {} --> weight: {}".format(i, self.weights[i]))
