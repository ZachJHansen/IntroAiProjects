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
from game import Directions
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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        #print(successorGameState)
        #print(newPos)
        #print(newFood)
        #for g in newGhostStates:
        #    print(str(g))

        fc = successorGameState.getNumFood()
        if fc != 0:
            fc = 1/fc
            #nf = self.nearest_food(newPos, newFood, fc)
        else:
            fc = 1
            #fc, nf = 0, 0
        if currentGameState.getPacmanPosition() == newPos:
            p = 1
        else:
            p = 0
        #print("nf", nf, "fc", fc)
        g1 = newGhostStates[0]
        weights = {"ghost_prox1" : 0.8, 
                    #"ghost_prox2": 0.3, 
                    "food_total": 0.9, 
                    "food_prox": 0.2,
                    "food_count": 0.9, 
                    "paralysis": -1, 
                    "score": 0.1,
                    "win": 10,
                    "lose": -10}
        features = {
            "ghost_prox1" : self.ghost_eval(newPos, g1, newScaredTimes[0]), 
            #"ghost_prox2" : tanh(DrManhattan(newPos, g2.getPosition()) - 1.75),
            "food_total": self.total_food_dist(newPos, newFood),
            #"food_prox": self.nearest_food(newPos, newFood, fc), 
            "food_count": fc,
            "paralysis": p,
            "score": successorGameState.getScore(),
            "win": successorGameState.isWin(),
            "lose": successorGameState.isLose()
        }
        
        if (len(newGhostStates)) > 1:
            g2 = newGhostStates[1]
            features["ghost_prox2"] = self.ghost_eval(newPos, g2, newScaredTimes[1])
            weights["ghost_prox2"] = 0.8
        #weights = {"food_prox": 1}
        #features = {"food_prox": nf}
        ws = lambda key : weights[key] * features[key]
        expected_value = sum([ws(key) for key in features])
        #print(nf, " | ", expected_value)
        return expected_value

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
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

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
