


# inference.py
# ------------
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


import itertools
import random
import busters
import game

from util import manhattanDistance, raiseNotDefined


class DiscreteDistribution(dict):
    """
    A DiscreteDistribution models belief distributions and weight distributions
    over a finite set of discrete keys.
    """
    def __getitem__(self, key):
        self.setdefault(key, 0)
        return dict.__getitem__(self, key)

    def copy(self):
        """
        Return a copy of the distribution.
        """
        return DiscreteDistribution(dict.copy(self))

    def argMax(self):
        """
        Return the key with the highest value.
        """
        if len(self.keys()) == 0:
            return None
        all = list(self.items())
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def total(self):
        """
        Return the sum of values for all keys.
        """
        return float(sum(self.values()))

    def normalize(self):
        """
        Normalize the distribution such that the total value of all keys sums
        to 1. The ratio of values for all keys will remain the same. In the case
        where the total value of the distribution is 0, do nothing.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> dist.normalize()
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
        >>> dist['e'] = 4
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
        >>> empty = DiscreteDistribution()
        >>> empty.normalize()
        >>> empty
        {}
        """
        "*** YOUR CODE HERE ***"
        #find the sum value of the distribution
        total = self.total()
        #if total 0
        if total == 0:
            #do nothing
            return
        #iterating over each key and accessing the value of the distribution
        for key in self.keys():
            #divide the value stored in the distribution by accessing with the help of key by the total
            newvalue= self[key] / total
            #update the value of the distribution
            self[key]=newvalue


    def sample(self):
        """
        Draw a random sample from the distribution and return the key, weighted
        by the values associated with each key.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> N = 100000.0
        >>> samples = [dist.sample() for _ in range(int(N))]
        >>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
        0.2
        >>> round(samples.count('b') * 1.0/N, 1)
        0.4
        >>> round(samples.count('c') * 1.0/N, 1)
        0.4
        >>> round(samples.count('d') * 1.0/N, 1)
        0.0
        """
        "*** YOUR CODE HERE ***"
        #find the sum of the values of the distribution
        total = self.total()
        #if total is 0
        if total == 0:
            #do nothing
            return
        #picking a random value from the distribution
        rand = random.random()
        temp = 0
        #iterating over each key and accessing the value of the distribution
        for key in self.keys():
            temp = temp + self[key]
            #if the random value selected is less than the value of the distribution accessed using the key when it divides the total
            if rand <= temp / total:
                #then return the key as intended by the question (Used to draw a sample from the distribution)
                return key
        raiseNotDefined()


class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    """
    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        """
        Set the ghost agent for later access.
        """
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = []  # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistributionHelper(self, gameState, pos, index, agent):
        try:
            jail = self.getJailPosition()
            gameState = self.setGhostPosition(gameState, pos, index + 1)
        except TypeError:
            jail = self.getJailPosition(index)
            gameState = self.setGhostPositions(gameState, pos)
        pacmanPosition = gameState.getPacmanPosition()
        ghostPosition = gameState.getGhostPosition(index + 1)  # The position you set
        dist = DiscreteDistribution()
        if pacmanPosition == ghostPosition:  # The ghost has been caught!
            dist[jail] = 1.0
            return dist
        pacmanSuccessorStates = game.Actions.getLegalNeighbors(pacmanPosition, \
                gameState.getWalls())  # Positions Pacman can move to
        if ghostPosition in pacmanSuccessorStates:  # Ghost could get caught
            mult = 1.0 / float(len(pacmanSuccessorStates))
            dist[jail] = mult
        else:
            mult = 0.0
        actionDist = agent.getDistribution(gameState)
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            if successorPosition in pacmanSuccessorStates:  # Ghost could get caught
                denom = float(len(actionDist))
                dist[jail] += prob * (1.0 / denom) * (1.0 - mult)
                dist[successorPosition] = prob * ((denom - 1.0) / denom) * (1.0 - mult)
            else:
                dist[successorPosition] = prob * (1.0 - mult)
        return dist

    def getPositionDistribution(self, gameState, pos, index=None, agent=None):
        """
        Return a distribution over successor positions of the ghost from the
        given gameState. You must first place the ghost in the gameState, using
        setGhostPosition below.
        """
        if index == None:
            index = self.index - 1
        if agent == None:
            agent = self.ghostAgent
        return self.getPositionDistributionHelper(gameState, pos, index, agent)

    def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition):
        """
        Return the probability P(noisyDistance | pacmanPosition, ghostPosition).
        """
        "*** YOUR CODE HERE ***"
        #first check
        #Check ghost is in the jail
        if ghostPosition==jailPosition:
            #initialise prob as 1 if noisydistance is 1 else 0
            prob=1 if (noisyDistance==None) else 0
            #return prob
            return prob
        #check if noisy distance is None
        if noisyDistance==None:
            #returns 0 as prob if true
            return 0
        #calculate the true distance using manhattan distance function as advised in the question
        trueDistance=manhattanDistance(pacmanPosition,ghostPosition)
        #find the prob given noisyDistance and true Distance
        Prob= busters.getObservationProbability(noisyDistance,trueDistance)
        #return the probability calculated
        return Prob
        #raiseNotDefined()

    def setGhostPosition(self, gameState, ghostPosition, index):
        """
        Set the position of the ghost for this inference module to the specified
        position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observe.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[index] = game.AgentState(conf, False)
        return gameState

    def setGhostPositions(self, gameState, ghostPositions):
        """
        Sets the position of all ghosts to the values in ghostPositions.
        """
        for index, pos in enumerate(ghostPositions):
            conf = game.Configuration(pos, game.Directions.STOP)
            gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
        return gameState

    def observe(self, gameState):
        """
        Collect the relevant noisy distance observation and pass it along.
        """
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index:  # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observeUpdate(obs, gameState)

    def initialize(self, gameState):
        """
        Initialize beliefs to a uniform distribution over all legal positions.
        """
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.allPositions = self.legalPositions + [self.getJailPosition()]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        """
        Set the belief state to a uniform prior belief over all positions.
        """
        raise NotImplementedError

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        raise NotImplementedError

    def elapseTime(self, gameState):
        """
        Predict beliefs for the next time step from a gameState.
        """
        raise NotImplementedError

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        raise NotImplementedError


class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward algorithm updates to
    compute the exact belief function at each time step.
    """
    def initializeUniformly(self, gameState):
        """
        Begin with a uniform distribution over legal ghost positions (i.e., not
        including the jail position).
        """
        self.beliefs = DiscreteDistribution()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        self.allPositions is a list of the possible ghost positions, including
        the jail position. You should only consider positions that are in
        self.allPositions.

        The update model is not entirely stationary: it may depend on Pacman's
        current position. However, this is not a problem, as Pacman's current
        position is known.
        """
        "*** YOUR CODE HERE ***"
        #Get Pacman Position
        pacPos = gameState.getPacmanPosition()
        #Get the jail Position
        jailPos = self.getJailPosition()
        #assign beliefs to a variable
        beliefsProb = self.beliefs
        #iterate through each postion
        for pos in self.allPositions:
            #calculate the probofobservation
            probofobservation= self.getObservationProb(observation, pacPos, pos, jailPos)
            #update the belifsProb given a pos
            beliefsProb[pos] =beliefsProb[pos] * probofobservation
        #Normalizes the updated Beliefs
        self.beliefs.normalize()

    def elapseTime(self, gameState):
        """
        Predict beliefs in response to a time step passing from the current
        state.

        The transition model is not entirely stationary: it may depend on
        Pacman's current position. However, this is not a problem, as Pacman's
        current position is known.
        """
        "*** YOUR CODE HERE ***"
        #Initialise the beliefs
        beliefsProb = self.beliefs
        #initialise a discreteDistribution
        newbeliefsProb = DiscreteDistribution()
        #Initialise a dict() to store the new position distribution
        newpositiondistribution = dict()

        #iterate through all the positions
        for pos in self.allPositions:
            #initialise values to the dictionary given the position
            newpositiondistribution[pos] = self.getPositionDistribution(gameState, pos)

        #iterate through all the positions
        for pos in self.allPositions:
            #iterate through prevpositions
            for prevpos in self.allPositions:
                #update the newbeliefsprob distribution based on current pos and previous pos
                newbeliefsProb[pos] +=newpositiondistribution[prevpos][pos] * beliefsProb[prevpos]
        #update the beliefs
        self.beliefs = newbeliefsProb
        #Normalize the beliefs
        self.beliefs.normalize()


    def getBeliefDistribution(self):
        return self.beliefs
        raiseNotDefined()

    def getBeliefDistribution(self):
        return self.beliefs


class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.
    """
    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent)
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initializeUniformly(self, gameState):
        """
        Initialize a list of particles. Use self.numParticles for the number of
        particles. Use self.legalPositions for the legal board positions where
        a particle could be located. Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior. Use
        self.particles for the list of particles.
        """
        self.particles = [] #list of particles
        "*** YOUR CODE HERE ***"
        # obtaining the number of particles
        numparticles = self.numParticles
        # while loop will execute unlit number of particle stays greater than zero
        while numparticles > 0:
            # if the number of particle is greater than the legal position length
            # then add the legal position to the list of particles
            # and the length of legal position is subtracted from the number of particles
            # this is done to achieve even distribution of particles across legal positions
            if numparticles > len(self.legalPositions):
                self.particles += self.legalPositions
                numparticles -= len(self.legalPositions)
            #if the number of particels is not greater than length of legal position then set number of particales to zero and exit the while loop
            else:
                self.particles += self.legalPositions[0:numparticles]
                numparticles = 0
        #raiseNotDefined()

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
        # obtaining pacman's position
        pacmanPosition = gameState.getPacmanPosition()
        # obtaining jail position
        jailPosition = self.getJailPosition()
        # obtaining discrete distribution
        distr = DiscreteDistribution()
        # iterating over all likely ghost position and finding their probability of the observation given the pacman's position, likely ghost position and jail position
        for ghostposition in self.particles:
            probability = self.getObservationProb(observation, pacmanPosition, ghostposition, jailPosition)
            distr[ghostposition] = distr[ghostposition] + probability
        # the cane in which the particle weight is not equal to zero
        if distr.total() != 0:
            # normalizing the distribution
            distr.normalize()
            self.beliefs = distr
            # implementing the sample method of the DiscreteDistribution class
            for n in range(self.numParticles):
                sample = distr.sample()
                self.particles[n] = sample
        # the case in which the particle has zero weight, the list of particles is reinitialized by calling the function initializeUniformly
        else:
            self.initializeUniformly(gameState)

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        "*** YOUR CODE HERE ***"
        # initialize a new list of particles
        new_list_of_particles = []
        # iterating over all the particle positions in self.sample
        for oldpos in self.particles:
            # it obtains the distribution over new positions for the ghost, given its previous position i.e. oldpos
            newPosDist = self.getPositionDistribution(gameState, oldpos)
            # using the sample method of the DiscreteDistribution class to get the key
            newpos = newPosDist.sample()
            # append the new position to the new list of particles
            new_list_of_particles.append(newpos)
        # assigning this new list of particles back to self.particles
        self.particles = new_list_of_particles

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution.

        This function should return a normalized distribution.
        """
        "*** YOUR CODE HERE ***"
        # storing discrete distribution to be used while converting to object
        distr = DiscreteDistribution()
        #then take the list of particles and convert it into a DiscreteDistribution object
        for p in self.particles:
            distr[p] = distr[p] + 1
        # normalizing the distribution
        distr.normalize()
        # returning the normalized distribution
        return distr
        raiseNotDefined()


class JointParticleFilter(ParticleFilter):
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """
    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def initialize(self, gameState, legalPositions):
        """
        Store information about the game, then initialize particles.
        """
        self.numGhosts = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeUniformly(gameState)

    def initializeUniformly(self, gameState):
        """
        Initialize particles to be consistent with a uniform prior. Particles
        should be evenly distributed across positions in order to ensure a
        uniform prior.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"
        #get the legal possitions
        possiblePos=self.legalPositions
        #used in the intertools function in repeat
        repeatcount=self.numGhosts
        #Using Intertools in python itertools.product(*iterables, repeat=1) where *iterables will be the legal Positions and repeat means how many times it multiplies with itself
        perm = itertools.product(possiblePos, repeat = repeatcount)
        #convert to list so we can iterate through it
        permutations=list(perm)
        #according to the question description we have to make it random
        random.shuffle(permutations)
        self.particles=permutations
        #raiseNotDefined()

    def addGhostAgent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1)

    def observe(self, gameState):
        """
        Resample the set of particles using the likelihood of the noisy
        observations.
        """
        observation = gameState.getNoisyGhostDistances()
        self.observeUpdate(observation, gameState)

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distances to all ghosts you
        are tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"

        #initialise pacman position
        pacPos = gameState.getPacmanPosition()
        #initialise num of ghosts
        n=self.numGhosts
        #initialise a empty list to store the jail positions
        jailPos= []
        #iterate through each ghost and store jail Position
        for posindex in range(n):
            jailPos.append(self.getJailPosition(posindex))
        #initialise the beliefs
        beliefsProb = self.getBeliefDistribution()
        #iterate over each particle in belief
        for particle in beliefsProb:
            for posindex in range(n):
                #update the beliefs using getObservationProb function
                beliefsProb[particle] =beliefsProb[particle]* self.getObservationProb(observation[posindex],pacPos,particle[posindex],jailPos[posindex])
        #if there is no beliefs in distribution
        if beliefsProb.total() == 0:
            #call initialize Uniformly function
            self.initializeUniformly(gameState)
        else:
            #make beliefsProb a list
            beliefsProbList=list(beliefsProb)
            #initialise d to count no of particles
            d=self.numParticles
            #initialise beliefsprob values to a list
            beliefsProbListval=list(beliefsProb.values())
            #store updated particles
            updatedparticles=random.choices(beliefsProbList, k=d,weights=beliefsProbListval)
            #update particles
            self.particles = updatedparticles

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        newParticles = []
        for oldParticle in self.particles:
            newParticle = list(oldParticle)  # A list of ghost positions

            # now loop through and update each entry in newParticle...
            "*** YOUR CODE HERE ***"
            #initialise the num of ghosts
            n=self.numGhosts
            for i in range(n):
                #newposdis function as described in question
                newPosDis = self.getPositionDistribution(gameState, oldParticle, i, self.ghostAgents[i])
                newParticle[i] = newPosDis.sample()

            """*** END YOUR CODE HERE ***"""
            newParticles.append(tuple(newParticle))
        self.particles = newParticles


# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()


class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """
    def initializeUniformly(self, gameState):
        """
        Set the belief state to an initial, prior value.
        """
        if self.index == 1:
            jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observe(self, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        if self.index == 1:
            jointInference.observe(gameState)

    def elapseTime(self, gameState):
        """
        Predict beliefs for a time step elapsing from a gameState.
        """
        if self.index == 1:
            jointInference.elapseTime(gameState)

    def getBeliefDistribution(self):
        """
        Return the marginal belief over a particular ghost by summing out the
        others.
        """
        jointDistribution = jointInference.getBeliefDistribution()
        dist = DiscreteDistribution()
        for t, prob in jointDistribution.items():
            dist[t[self.index - 1]] += prob
        return dist
