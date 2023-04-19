"""
    Student Team Info:
        1)  Name: Abhijeet Banerjee
	        GMU ID: G01349260 
        2)  Name : Joel Sadanand Samson
            GMU ID: G01352483  
         
"""




# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    #Initialize ‘current’ node to start state
    #Used to signify Current state (also in problems)
    currentNode = problem.getStartState()
    #Initialize ‘closed’ as an empty list
    #Used to signify the Visited Nodes
    ClosedList = []
    #Initialize ‘open’ as one of (stack, queue, priority queue) 
    #For Dfs we use stack
    #Signifies Open in the Problems
    OpenStack = util.Stack()
    #Create a Node tuple with elements as nodes in the form State,Action
    NodeTuple = (currentNode, [])
    #Push the Node tuple to the stack
    OpenStack.push(NodeTuple)
    
    #Checks if the stack is empty or not using while
    while not OpenStack.isEmpty():
        #Pop first element of the stack that has info as state,action
        Nodecurr = OpenStack.pop()
        #assign the state  value to a variable called "state"
        state=Nodecurr[0]
        #assign the action value to a variable called "posdir"
        posdir=Nodecurr[1]
        #append the state to the closed list to incdicate that we have visited the state
        ClosedList.append(state)
        #check if the goal state has been reached using the function .isGoalState(state) from stack functions in util.Stack
        if problem.isGoalState(state):
                #if the goal has been reached then return the action as the same value popped from the stack
                return posdir
        else:
                #used to get the successor nodes using the fuction .getSuccessors(state)
                successors = problem.getSuccessors(state)
                #According to the algorithum loop accross the succssors 
                #Each element in the successor that is accessed using variable succ has values in the form state,action
                for succ in successors:
                    #Assign the value of the state to a variable "currentState" 
                    currentState=succ[0]
                    #Check if the currentState is not in the ClosedList (Visited)
                    if not currentState in ClosedList:
                        #assign the state to currentNode
                        currentNode = succ[0]
                        #assign the posdir(action) to posaction
                        posaction = succ[1]
                        #push the new value to the stack with the action as the sum of previous plus the new
                        OpenStack.push((currentNode, posdir + [posaction]))
    #returns the actions to be taken                    
    return posdir  
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #Initialize ‘current’ node to start state
    #Used to signify Current state (also in problems)
    currentNode = problem.getStartState()
    #Initialize ‘closed’ as an empty list
    #Used to signify the Visited Nodes
    ClosedList = []
    #Initialize ‘open’ as one of (stack, queue, priority queue) 
    #For Bfs we use Queue
    #Signifies Open in the Problems
    OpenQueue = util.Queue()
    #Create a Node tuple with elements as nodes in the form State,Action
    NodeTuple = (currentNode, []) 
    #Push the Node tuple to the Queue
    OpenQueue.push(NodeTuple) 
    
    #Checks if the Queue is empty or not using while
    while not OpenQueue.isEmpty():
        #Pop first element of the Queue that has info as state,action
        Nodecurr = OpenQueue.pop()
        #assign the state  value to a variable called "state"
        state=Nodecurr[0]
        #assign the action value to a variable called "posdir"
        posdir=Nodecurr[1]
        #Check if state popped from Queue is not in ClosedList ie. Visited
        if state not in ClosedList:
            #If not then push to Closed List 
            #used to signify that it has been visited
            ClosedList.append(state)
            
            #check if the goal state has been reached using the function .isGoalState(state) from Queue functions in util.Queue
            if problem.isGoalState(state):
                #if the goal has been reached then return the action as the same value popped from the stack
                return posdir
            
            else:
                #used to get the successor nodes using the fuction .getSuccessors(state) 
                successors = problem.getSuccessors(state)
                #According to the algorithum loop accross the succssors 
                #Each element in the successor that is accessed using variable succ has values in the form state,action
                for succ in successors:
                     #Assign the value of the state to a variable "currentState" 
                    currentState=succ[0]
                     #Check if the currentState is not in the ClosedList (Visited)
                    if not currentState in ClosedList:
                        currentNode = succ[0]
                        #assign the posdir(action) to posaction
                        posaction = succ[1]
                        #push the new value to the stack with the action as the sum of previous plus the new
                        OpenQueue.push((currentNode, posdir + [posaction]))
    #returns the actions to be taken     
    return posdir 

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    #Initialize ‘current’ node to start state
    #Used to signify Current state (also in problems)
    currentNode = problem.getStartState()
    #Initialize ‘closed’ as an empty list
    #Used to signify the Visited Nodes
    ClosedList = []
    #Initialize ‘open’ as one of (stack, queue, priority queue) 
    #For Unoform Search Cost we use PriorityQueue
    #Signifies Open in the Problems
    OpenPQueue = util.PriorityQueue()
    #Create a Node tuple with elements as nodes in the form State,Action
    NodeTuple=(currentNode, [])
    #Push the Node tuple to the Priority Queue 
    #Here Each elemnent in the PQueue Contains values of the form State, Action ,Cost
    OpenPQueue.push((NodeTuple) ,0)
    
    #Checks if the PriorityQueue is empty or not using while
    while not OpenPQueue.isEmpty():
        #Pop first element of the Queue that has info as state,action
        Nodecurr = OpenPQueue.pop()
        #assign the state  value to a variable called "state"
        state=Nodecurr[0]
        #assign the action value to a variable called "Posdir"
        posdir=Nodecurr[1]
        #check if the goal state has been reached using the function .isGoalState(state) from Queue functions in util.Queue
        if problem.isGoalState(state):
            #if the goal has been reached then return the action as the same value popped from the stack
            return posdir
        #Check if state popped from PQueue is not in ClosedList ie. Visited
        if state not in ClosedList:
            #If not then push to Closed List 
            #used to signify that it has been visited
            ClosedList.append(state)
             #used to get the successor nodes using the fuction .getSuccessors(state) 
            successors = problem.getSuccessors(state)
            #According to the algorithum loop accross the succssors 
            #Each element in the successor that is accessed using variable succ has values in the form state,action,cost
            for succ in successors:
                #Assign the value of the state to a variable "currentState" 
                currentState = succ[0]
                if currentState not in ClosedList:
                    #Assign the value of the state to a variable "currentState" 
                    currentNode=succ[0]
                     #assign the posdir(action) to posaction
                    posaction = succ[1]
                    #new cost ie. The cost of reaching here from the previous nodes is calculated
                    newCost = posdir + [posaction]
                    #push the new PQueue to the stack with the action as the sum of previous plus the new
                    OpenPQueue.push((currentState, posdir + [posaction]), problem.getCostOfActions(newCost))
    #returns the actions to be taken     
    return posdir
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    #Initialize ‘current’ node to start state
    #Used to signify Current state (also in problems)
    currentNode = problem.getStartState()
    #Initialize ‘closed’ as an empty list
    #Used to signify the Visited Nodes
    ClosedList = []
    #Initialize ‘open’ as one of (stack, queue, priority queue) 
    #For a* we use PriorityQueue
    #Signifies Open in the Problems
    OpenPQueue = util.PriorityQueue()
    #Create a Node tuple with elements as nodes in the form State,Action
    NodeTuple=(currentNode, []) 
    #Push the Node tuple to the Priority Queue 
    #Here Each elemnent in the PQueue Contains values of the form State, Action ,Cost
    OpenPQueue.push(NodeTuple,nullHeuristic(currentNode,problem))
    
    #Checks if the PriorityQueue is empty or not using while
    while not OpenPQueue.isEmpty():
        #Pop first element of the Queue that has info as state,action
        Nodecurr = OpenPQueue.pop()
        #assign the state  value to a variable called "state"
        state=Nodecurr[0]
        #assign the action value to a variable called "Posdir"
        Posdir=Nodecurr[1]
        #Check if state popped from PriorityQuueue is not in ClosedList ie. Visited
        if state not in ClosedList:
            #If not then push to Closed List 
            #used to signify that it has been visited
            ClosedList.append(state)
            #check if the goal state has been reached using the function .isGoalState(state) from Queue functions in util.Queue
            if problem.isGoalState(state):
            #if the goal has been reached then return the action as the same value popped from the stack
                return Posdir
            else:
        #used to get the successor nodes using the fuction .getSuccessors(state) 
                successors = problem.getSuccessors(state)
            #According to the algorithum loop accross the succssors 
            #Each element in the successor that is accessed using variable succ has values in the form state,action
                for succ in successors:
                     #Assign the value of the state to a variable "currentState" 
                    currentState=succ[0]
                     #Check if the currentState is not in the ClosedList (Visited)
                    if not currentState in ClosedList:
                        currentNode = succ[0]
                        #assign the posdir(action) to posaction
                        posaction = succ[1]
                        #push the new PQueue to the stack with the action as the sum of previous plus the new
                        newCost = problem.getCostOfActions(Posdir + [posaction]) + heuristic(currentState, problem)
                        #push the new value to the stack with the action as the sum of previous plus the new
                        OpenPQueue.push((currentState, Posdir + [posaction]), newCost)
    #returns the actions to be taken     
    return Posdir 



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
