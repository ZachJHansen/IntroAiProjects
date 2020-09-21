# multiAgents.py
# --------------
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


from util import manhattanDistance as DrManhattan
from game import Directions, Actions
from math import tanh
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    # Returns the cumulative distance between pacman and remaining food
    # Randomness introduced to remove thrashing
    def total_food_dist(self, pacman, food_grid):
        total = 0
        master = []
        c = 1
        for col in food_grid:
            indices = [(indx, c) for indx in range(len(col)-1, 1, -1) if col[indx] == True]
            master = master + indices
            c += 1
        for pellet in master:
            total += DrManhattan(pacman, pellet) #+ random.randrange(1,3)
        if total == 0:
            return 1
        else:
            return 1/total

    # Returns distance to nearest food (is this working right?)
    def nearest_food(self, pacman, food_grid, food_count):
        master = []
        c = 1
        for col in food_grid:
            indices = [(indx, c) for indx in range(len(col)-1, 1, -1) if col[indx] == True]
            master = master + indices
            c += 1
        nearest = None
        nearest = (float('inf'), float('inf'))
        dist = float('inf')
        for food in master:
            candidate = DrManhattan(pacman, food)
            if candidate < dist:
                nearest = food
                dist = candidate
        #print(master)
        #print("Pacman", pacman, "nearest", nearest, "dist", dist)
        del master
        if dist == 0:
            return 1
        else:
            return 1/dist

    def ghost_eval(self, pacman, ghost, scared_time=0):
        tan = tanh(DrManhattan(pacman, ghost.getPosition()) - 1.75)
        if scared_time - 3 > DrManhattan(pacman, ghost.getPosition()):  # Bust the ghost!
            return 1/tan
        else:
            return tan              # Fly, you fools!

    # We hates thrashing
    def thrash_check(self, past_action, proposed_action):
        if past_action == Directions.STOP:                  # You should move
            return 1   
        else:
            reverse = Actions.reverseDirection(proposed_action)
            if (reverse == past_action):                            # You should not take proposed action
                return -1
            else:                               # Take the proposed action
                return 1

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    # An evaluation function is a weighted linear function of features used to estimate utility of a state
    # Possible Pacman features: ghost proximity, food proximity, food count
    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        history = currentGameState.getPacmanState().getDirection()
        successorGameState = currentGameState.generatePacmanSuccessor(action)        
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


        "*** YOUR CODE HERE ***"
        distanceOfFoods = []  # create a list holding each manHattanDistance from currentPos to foods position
        value = 0

        for food in newFood.asList():  # given a list of new Food positions add their collection of distances
            distanceOfFoods.append(DrManhattan(newPos, food))  # add to list of distances of each food item

        for distance in distanceOfFoods:  # for each
            if distance >= 15:  # if the food is far away not as highly valued
                value += .20
            elif distance < 15 and distance > 4:  # if its generally medium distance its a bit better to take
                value += .50
            else:
                value += 1  # if its really close, its valued as the best possible food to eat
        ghostPositions = successorGameState.getGhostPositions()

        for ghost in ghostPositions:  # for each ghost in ghostPositions find pacmans current distance away from them
            currentDistance = self.evaluateGhostvsPacPositions(ghost, newPos)
            if currentDistance == 0:  # if ghost is ontop of pacman huge failure/lost
                value -= 3
            elif currentDistance < 4:  # if ghost is relatively close deduct the value
                value -= 2
        finalEvalutaion = value + successorGameState.getScore()  # get the finalEvaluationValue

        return finalEvalutaion

    def evaluateGhostvsPacPositions(self,ghost,pacman):
        distanceBetweenPacManAndGhost = DrManhattan(pacman, ghost)
        # print("This is the distance between pacman and ghost: ", distanceBetweenPacManAndGhost)
        return distanceBetweenPacManAndGhost


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        num_agents = gameState.getNumAgents()
        max_depth = (self.depth * num_agents) - 1           # Ply * agents per ply (0 indexed)
        current_depth = 0
        best_action, best_score = None, float('-inf')
        for action in gameState.getLegalActions(0):#self.index):
            score = self.value(gameState.generateSuccessor(self.index, action), num_agents, current_depth+1, max_depth)
            if score > best_score:
                best_action, best_score = action, score
        return best_action

    def max_value(self, state, num_agents, current_depth, max_depth):
        val = float('-inf')
        agentIndex = current_depth % num_agents
        actions = state.getLegalActions(agentIndex)
        for a in actions:
            successor = state.generateSuccessor(agentIndex, a)
            val = max(val, self.value(successor, num_agents, current_depth+1, max_depth))
        return val

    def min_value(self, state, num_agents, current_depth, max_depth):
        val = float('inf')
        agentIndex = current_depth % num_agents
        actions = state.getLegalActions(agentIndex)
        for a in actions:
            successor = state.generateSuccessor(agentIndex, a)
            val = min(val, self.value(successor, num_agents, current_depth+1, max_depth))
        return val

    # Returns true if the state is terminal (win or loss) or a leaf node at depth D
    def terminal(self, state, current_depth, max_depth):
        if current_depth > max_depth or (state.isWin() or state.isLose()):
            return True
        else:
            return False

    # Input: Game State, number of agents, current tree depth (0 indexed), max tree depth (NOT the ply, which is supplied via command line)
    # Output: Recursively calculated minimax value for the provided state
    def value(self, state, num_agents, current_depth, max_depth):
        if self.terminal(state, current_depth, max_depth):
            return self.evaluationFunction(state)
        if current_depth % num_agents == 0:                                 # Is the next agent min or max?
            return self.max_value(state, num_agents, current_depth, max_depth)  # Max agent returns max of its childrens' values
        else:
            return self.min_value(state, num_agents, current_depth, max_depth)  # Min agent returns min of its childrens' values

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        num_agents = gameState.getNumAgents()
        max_depth = (self.depth * num_agents) - 1           # Ply * agents per ply (0 indexed)
        current_depth = 0
        best_action, best_score = None, float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        for action in gameState.getLegalActions(self.index):
            score = self.value(gameState.generateSuccessor(self.index, action), num_agents, current_depth+1, max_depth, alpha, beta)
            if score > best_score:
                best_action, best_score = action, score
            alpha = max(alpha, score)
        return best_action

    def max_value(self, state, num_agents, current_depth, max_depth, alpha, beta):
        val = float('-inf')
        agentIndex = current_depth % num_agents
        actions = state.getLegalActions(agentIndex)
        for a in actions:
            successor = state.generateSuccessor(agentIndex, a)
            val = max(val, self.value(successor, num_agents, current_depth+1, max_depth, alpha, beta))
            if val > beta:        # The current node knows the maximizer above it will never choose a value > beta so it terminates recursion early
                return val
            alpha = max(alpha, val)
        return val

    def min_value(self, state, num_agents, current_depth, max_depth, alpha, beta):
        val = float('inf')
        agentIndex = current_depth % num_agents
        actions = state.getLegalActions(agentIndex)
        for a in actions:
            successor = state.generateSuccessor(agentIndex, a)
            val = min(val, self.value(successor, num_agents, current_depth+1, max_depth, alpha, beta))
            if val < alpha:        # The current node knows the maximizer above it will never choose a value < alpha so it terminates recursion early
                return val
            beta = min(beta, val)
        return val

    # Returns true if the state is terminal (win or loss) or a leaf node at depth D
    def terminal(self, state, current_depth, max_depth):
        if current_depth > max_depth or (state.isWin() or state.isLose()):
            return True
        else:
            return False

    # Input: Game State, number of agents, current tree depth (0 indexed), max tree depth (NOT the ply, which is supplied via command line)
    # Output: Recursively calculated minimax value for the provided state with alpha-beta pruning
    def value(self, state, num_agents, current_depth, max_depth, alpha, beta):
        if self.terminal(state, current_depth, max_depth):
            return self.evaluationFunction(state)
        if current_depth % num_agents == 0:                                                  # Is the next agent min or max?
            return self.max_value(state, num_agents, current_depth, max_depth, alpha, beta)  # Max agent returns max of its childrens' values
        else:
            return self.min_value(state, num_agents, current_depth, max_depth, alpha, beta)  # Min agent returns min of its childrens' values

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
