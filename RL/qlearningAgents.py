"""
    Student Team Info:
        1)  Name: Abhijeet Banerjee
	          Gnum: G01349260
        2)  Name: Joel Sadanand Samson
            Gnum: G01352483
        

"""
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


from xml.sax.handler import feature_external_ges
from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math
import math
import sys

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
        self.values = util.Counter()
        "*** YOUR CODE HERE ***"

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        #If we seen a state
        if self.getLegalActions(state):
              #return counter dict that stores the Qvalue of state given action
              return self.values[(state,action)]
        else:
              #if we have never seen a state
              return 0.0
        util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        #get actions that can be performed in a state
        actions = self.getLegalActions(state)
        #calculate length of actions
        i=len(actions)
        #assign max qvalue to compare (it has to be a large negative value as qvalue can be negative also)
        maxvalue=-999999999999999
        #if length not equal for zero that is legal actions are present for state not terminating
        if(i!=0):
          #iterate over each action in list of actions
          for action in actions:
                #get qValue of each state given action
                qvalue=self.getQValue(state,action)
                #compare if qvalue is greater than maxvalue
                if(maxvalue<=qvalue):
                    #then assign maxvalue as the q Value
                    maxvalue=qvalue
          #above for loop used to get the max q value
          #return the Qvalue
          return maxvalue
        else:
          #if terminal state return q value as 0
          return 0.0
        util.raiseNotDefined()


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        #get legal actions over state
        actions = self.getLegalActions(state)
        #Calculate q value of state
        Qvalue = self.computeValueFromQValues(state)
        #calculate no of actions
        i=len(actions)
        #assign a list list to store actions
        randaction = []
        #check if not a terminal state that is length of actions not zero
        if(i!=0):
          #iterate through each action in actions
          for action in actions:
                #get its qvalue in the state for particular action
            Qvalueforaction=self.getQValue(state,action)
            #compare the qvalues
            if Qvalueforaction== Qvalue:
                #if true append to list
                randaction.append(action)
        else:
              #if terminal state no actions in legal action so length is zero
              return None
        #according to question we have to break ties randomly for better behavior
        #so using random.choice given in question choose a random action in list randaction and return
        return random.choice(randaction)
        util.raiseNotDefined()


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
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        #we have to choose random action epsilon times
        #if len of actions is zero ie. terminal state return action as None(no action)
        if len(legalActions)==0:
              return None
        #if we flip a coin and it returns true
        if util.flipCoin(self.epsilon) == True:
          #choose a random action from list of legal actions
          action = random.choice(legalActions)
        else:
          #else compute the action for the state using function computeActionFromQvalues
          action = self.computeActionFromQValues(state)
        #returns the action
        return action
        util.raiseNotDefined()



    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        #get q value of current state
        qValueState = self.getQValue(state,action)
        #get q value of next state
        qValueNextState = self.computeValueFromQValues(nextState)
        #update the qvalues stores in the counter dic values that stores q value of a state and particular action
        self.values[(state,action)] = qValueState + self.alpha*(reward + self.discount*(qValueNextState)- qValueState)

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
        #initialise Qvalue as 0 first
        qvalue=0
        #extract features with states and action
        feature=self.featExtractor.getFeatures(state,action)
        #for each feature
        for f in feature:
            #Extract the weights
            weight=self.weights[f]
            #Extract feature vectors (state,action)
            featureVector=feature[f]
            #calculate the Qvalue based on formula given in Question 10
            qvalue=qvalue+weight * featureVector
        #return Qvalue
        return qvalue
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        #intialise the discount
        d=self.discount
        #Initialise Qvalue for current state
        q1value=self.getQValue(state,action)
        #Initialise Qvalue for nextstate
        q2value=self.getValue(nextState)
        #Calculate the difference term using the fourmula given in the Question 10
        diff = reward + d*q2value - q1value
        #Extract features using state and action
        feature=self.featExtractor.getFeatures(state,action)
        #for each feature
        for f in feature:
            #intialise weights
            weight=self.weights[f]
            #intialise alpha
            alpa=self.alpha
            #intialise feature vector (state,action)
            featureVector=feature[f]
            #update value if weights using formula in the Question 10
            self.weights[f]=weight+alpa*diff*featureVector




    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
