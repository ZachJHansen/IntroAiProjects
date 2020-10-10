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
              mdp.getStates() --> returns tuple (x,y) cordinates
              mdp.getPossibleActions(state) ---> returns a tuple
              mdp.getTransitionStatesAndProbs(state, action) --> returns a tuple with ( (x,y), # ) --> ( state, probability)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount #Gamma
        self.iterations = iterations # i
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        #Define Reward

        for i in range (self.iterations):
            #Hint: Use the util.Counter class in util.py, which is a dictionary with a default value of zero.
            # Methods such as totalCount should simplify your code.
            # However, be careful with argMax: the actual argmax you want may be a key not in the counter!
            #copy what was in the counter (This is V* in the end.)
            vSOptimal = self.values.copy()
            #Loop through all the available states in the MDP
            for state in self.mdp.getStates():
                #Set the current max value to -inf and update towards +inf (Just like in minimax)
                maxV = float("-inf")
                #Get the available actions for the current state
                for action in self.mdp.getPossibleActions(state):
                    #initialize the value of the current state value to zero.
                    vS = 0
                    #Derive the transition info
                    for transition in self.mdp.getTransitionStatesAndProbs(state, action):
                        # Get the value of the state we land in (V(s')) and Transition probability to next state (T)
                        vSPrime, t = transition
                        # Get the reward for the landing state (R)
                        r = self.mdp.getReward(state, action, vSPrime)
                        #update the current value with Bellman eqn.
                        vS += t* (r + self.discount * self.values[vSPrime])
                    #Check if you are in a terminal state, if not, optimal would be the max value.
                    if self.mdp.isTerminal(state) == False:
                        # update the max value if the current max is lower than the newfound value.
                        if vS > maxV:
                            maxV = vS
                         # Update V*(s)
                        vSOptimal[state] = maxV
                    #If you are, just take the value that was just calculated and exit.
                    else:
                        #Update V*(s)
                        vSOptimal[state] = vS
            self.values = vSOptimal


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        #Initialize the Q value to 0
        qValue = 0
        #Get the transition info
        for transition in self.mdp.getTransitionStatesAndProbs(state, action):
            # Get the value of the state we land in (V(s')) and Transition probability to next state (T)
            vSPrime, t = transition
            #Get the reward for the landing state (R)
            r = self.mdp.getReward(state, action, vSPrime)
            #Apply Bellman's eqn to update the Q value.
            qValue = qValue + t * (r + self.discount * self.values[vSPrime])
        return qValue


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #Get the valid actions at the current state
        possibleActions = self.mdp.getPossibleActions(state)
        #Check if we have reached a terminal state. If you have, take the reward and exit in the next time step.
        if self.mdp.isTerminal(state) == False:
            #Initialize optimal action
            optimalAction = possibleActions[0]
            #Get the Q value for the current optimal action, (Initializing).
            qOptimal = self.getQValue(state, optimalAction)
            #Iterate through the available actions.
            for action in possibleActions:
                #Get the Q value for current action.
                qValue = self.getQValue(state, action)
                if qValue > qOptimal:
                    # Update optimal Q value if the newfound Q value is larger.
                    qOptimal = qValue
                    #Also, update the corresponding action (Pi(s)) by using argmax Bellman eqn.
                    optimalAction = action
            #Return the optimal action.
            return optimalAction
        else:
            return None


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
        # Runs cyclic value iteration: on each iteration, the value of a single state is updated
        key = 0
        for i in range(self.iterations):
            # Hint: Use the util.Counter class in util.py, which is a dictionary with a default value of zero.
            # Methods such as totalCount should simplify your code.
            # However, be careful with argMax: the actual argmax you want may be a key not in the counter!
            if key >= len(self.mdp.getStates()):
                key = 0
            state = self.mdp.getStates()[key]
            if self.mdp.isTerminal(state):
                key += 1
                continue
            else:
                val_list = []
                for action in self.mdp.getPossibleActions(state):
                    currentVal = self.computeQValueFromValues(state, action)
                    val_list.append(currentVal)
                self.values[state] = max(val_list)
            key += 1


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
        pred_dict = {}
        for s in self.mdp.getStates():
            pred_dict[s] = self.predecessors(s)
        priority_queue = util.PriorityQueue()
        # For non-terminal states s:
        for s in self.mdp.getStates():
            if self.mdp.isTerminal(s):
                continue
            else:
                qvals = []
                for action in self.mdp.getPossibleActions(s):
                    qvals.append(self.computeQValueFromValues(s, action))
                max_qval = max(qvals)
                diff = abs(self.values[s] - max_qval)
                priority_queue.push(s, (-1)*diff)
    # part 2
        for i in range(self.iterations):
            if priority_queue.isEmpty():
                break
            else:
                state = priority_queue.pop()
                qvals = []
                for action in self.mdp.getPossibleActions(state):
                    qvals.append(self.computeQValueFromValues(state,action))
                self.values[state] = max(qvals)

                for predessor in pred_dict[state]:
                    val_list = []
                    for action in self.mdp.getPossibleActions(predessor):
                        current_value = self.computeQValueFromValues(predessor, action)
                        val_list.append(current_value)
                    diff = abs(self.values[predessor] - max(val_list))
                    if diff > self.theta:
                        priority_queue.update(predessor, -diff)


    def predecessors(self, orig_state):
        preds = set()  # Init empty set
        for state_prime in self.mdp.getStates():  # All states are possible predecessors of orig_state
            escape = False
            if self.mdp.isTerminal(state_prime):#  or state_prime == orig_state:  # Except terminal states and orig_state itself
                continue
            for action in self.mdp.getPossibleActions(state_prime):
                for transition in self.mdp.getTransitionStatesAndProbs(state_prime, action):
                    candidate, Pt = transition  # Pt = P(Candidate | state_prime, action)
                    if Pt > 0 and candidate == orig_state:  # Non-zero prob for (state_prime, action, orig_state) sequence
                        preds.add(state_prime)
                        escape = True  # We know state_prime is a predecessor, so break inner loops
                        break
                if escape:
                    break
        return preds

