# dice-game
Dice Game simulator with AI that aims to maximise points. Project for MSc with Bath Uni.

# Introduction

This dice game consists of states, actions, and rewards (or punishments). We can therefore model it as 
a Markov Decision Process [REF].

# Extremely Simple Agents

Two agents were provided with the exercise material. I have placed their code in agents/bad_agents.py.
1) AlwaysHoldAgent: Will always hold given any dice state
2) PerfectionistAgent: Will just keep rolling until the dice provides a perfect score

These agents do not perform very well, but do create a base line of performance. 

They had the following averages over 100 games:

| Agent                 | Mean Score |
|-----------------------|------------|
| AlwaysHoldAgent       |  10.82     |
| PerfectionistAgent    | -41.80     |


# One Step Look Ahead

One method of evaluating the next best move in a given state is to look at all possible
outcomes for all actions you can take from that state.

You can calculate the resulting scores for all outcomes and the probability for
getting those scores. If, for a given action, the probability of a better outcome
is higher than the probability of a same or worse outcome, take that action.

This is a fairly simple algorithm and has been implemented in agents/one_step_look_ahead.py.

You can adjust how risky the agent is by changing the risk factor. The basic risk factor I chose
was 50%. If the probability of an action giving a better outcome is > 50%, it will take the action. 
From this I have also written a cautious agent (will only take an action if there's > 60% 
chance of success) and a risky agent (will take an action if there's > 40% chance of success).

These agents tend to perform OK with the following average results over 100 games:

| Agent                 | Mean Score |
|-----------------------|------------|
| OneStepLookAheadAgent |  12.33     |
| RiskyAgent            |  12.33     |
| CautiousAgent         |  12.15     |

All one step look ahead agents perform better than the extremely simple agents.
However, the cautious agent under performs the other one step look aheads. 

# Value Iteration

As this game can be modeled as a Markov Decision Process, there are a number of algorithms we can use
to create a policy set that performs better than looking ahead one step.

For this we will use the Bellman equations:








| Agent                              | Mean Score |
|------------------------------------|------------|
| MarkovDecisionProcessAgent         |  13.53     |
| MarkovDecisionProcessAgentAdjusted |  13.53     |

# Value Iteration - Adjusted


# Optimising Gamma and Theta


# Comparison of performance

![All Agents Except Perfectionist Performance](data/all_no_perf.png)

![Perfectionist Agent Performance](data/perf.png)
