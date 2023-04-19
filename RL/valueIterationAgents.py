"""
    Student Team Info:
        1)  Name: Abhijeet Banerjee
	        Gnum: G01349260
        2)  Name: Joel Sadanand Samson
            Gnum: G01352483
        

"""

# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util
import sys

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()




    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        #initialise iterations
        range_ctrl=self.iterations
        #initialise count
        i=0
        #Access the states
        states=self.mdp.getStates()
        while i in range(range_ctrl):
        #Use util.Counter to Initialise a Counter(List) to store values at each state.
          counter = util.Counter()
          #increment counter so that it can iterate over the range of iterations
          i=i+1


          #access each state s in states
          for s in states:
            #Condition to check if it is a terminal state
            if self.mdp.isTerminal(s):
                #if true then action will be to exit the program and nextstate will be nothing ie. ''
                action='exit'
                nxt=''
                #assign Reward at the the terminal state
                self.values[s] = self.mdp.getReward(s,action,nxt)
            # if the state is non-terminal, then finding the best value as the maximum of expected sum of rewards of different actions.
            else:
                #if non terminal find the best value of ie. max of rewards(expected) for different actions (value Iteration)
                #Get actions that can be done in that state
              step = self.getAction(s)
              #Compute q Value at that state
              counter[s] = self.computeQValueFromValues(s,step)
        #assign the whole counter dict to values counter dict
          self.values = counter



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        #returns value at the particular state
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        #Get the transitions states and probability
        tans = self.mdp.getTransitionStatesAndProbs(state, action)
        #assign value as 0 first
        val = 0
        #value calculated as discounted value of transition state + reward of getting transition
        #Adding these transition values gives the q-value for a state action pair
        #accessing each transition state
        for ts in tans:
            #assigning the current state as it will be stored first
            current=ts[0]
            #assigning next state as stored in the next value
            nxt=ts[1]
            #get current state reward
            reward = self.mdp.getReward(state,action,current)
            #use value iteration thought in class
            val = val + reward + self.discount*(self.values[current]*nxt)
        #return the value ie Qvalue
        return val
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #checking condition if the state is terminal return no actions
        if self.mdp.isTerminal(state):
            return None
        #if not then create a counter dict to store the Q values given a particular actions and state
        counter=util.Counter()
        #get the possibles action in that state
        action=self.mdp.getPossibleActions(state)
        #iterate over all possible actions
        for s in action:
            #compute the q value given state and action
            #store that q value in Counter dict
            counter[s]=self.computeQValueFromValues(state,s)
        #Access and store the max Qvalue in the counter dict in val
        val = max(counter,key=counter.get)
        #return val
        return val

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        #Access the states
        states = self.mdp.getStates()
        #Initialize iterations to be used in loops
        range_ctrl = self.iterations
        #A for loop to iterate over all statements in it and it is controlled by the range_ctrl variable
        for n in range(range_ctrl):
            st = states[n%len(states)]
            #Check if current state is not terminal
            # if not then assign action associated to state to step variable and assign Q value associated to the state and action to val variable
            if not self.mdp.isTerminal(st):
                step = self.getAction(st)
                val = self.getQValue(st, step)
                #store this new value
                self.values[st] = val
class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

        def runValueIteration(self):
            "*** YOUR CODE HERE ***"
        #Access states
        states = self.mdp.getStates()
        #Access priority queue
        pqueue = util.PriorityQueue()
        #Initialize predecessor
        predecessors = {}
        #This gives us the state previous to the current state
        for st in states:
            self.values[st] = 0
            #Initialize a list to store previous states
            prelist = []
            #A list that stores direction of movements
            direc = ['north', 'south', 'east', 'west']
            #A loop to consider each state
            for s in states:
                #Check if the is terminal or not
                if not self.mdp.isTerminal(s):
                    #if not terminal
                    for d in direc:
                        #Take each direction and check if its present in the action or not
                        if d in self.mdp.getPossibleActions(s):
                            #if true than access transition state and probability for that particular state and direction and store it in tans variable
                            tans = self.mdp.getTransitionStatesAndProbs(s, d)
                            #Access the next state from the transition state and probability and check if it is a part of states
                            for next, p in tans:
                                if (next == st):
                                    #Add it to the list
                                    prelist.append(s)
            #Assign the list of previous states to predecessor
            predecessors[st] = prelist
        range_ctrl = self.iterations
        for st in states:
            if not self.mdp.isTerminal(st):
                #This gives the best Q value amonsgt the possible actions
                qmax = -999999999
                for step in self.mdp.getPossibleActions(st):
                    if qmax < self.getQValue(st, step):
                        qmax = self.getQValue(st, step)
                #Take the difference of the best q value and value of the state
                sub = abs(self.values[st]-qmax)
                #Push it in the priority queue for that state
                pqueue.push(st, -sub)
        #A for loop to iterate over all statements in it and it is controlled by the range_ctrl variable
        for n in range(range_ctrl):
            if pqueue.isEmpty():
                return
            #the last element in the priority queue
            i = pqueue.pop()
            #This gives the best Q value amonsgt the possible actions
            qmax = -999999999
            for step in self.mdp.getPossibleActions(i):
                if qmax < self.getQValue(i, step):
                    qmax = self.getQValue(i, step)
            #Replace value for last state with max
            self.values[i] = qmax
            #a loop to iterate over the predecessors in the list
            for pre in predecessors[i]:
                #This gives the best Q value amonsgt the possible actions
                qmax = -999999999
                for step in self.mdp.getPossibleActions(pre):
                    if qmax < self.getQValue(pre, step):
                        qmax = self.getQValue(pre, step)
                #Take the difference of the best q value and value of the state
                sub = abs(self.values[pre] - qmax)
                #if difference is more than the value of theta
                # #then update the queue with that value fot that perticular predecessor state
                if sub > self.theta:
                    pqueue.update(pre, -sub)
