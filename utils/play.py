from agents import *
from dice_game import DiceGame
import numpy as np


def play_game_with_agent(agent, game, verbose=False):
    state = game.reset()

    if verbose:
        print(f"Testing agent: \n\t{type(agent).__name__}")
        print(f"Starting dice: \n\t{state}\n")

    game_over = False
    actions = 0
    while not game_over:
        action = agent.play(state)
        actions += 1
        if verbose:
            print(f"Action {actions}: \t{action}")
        penalty, state, game_over = game.roll(action)
        if verbose and (not game_over):
            print(f"Dice: \t\t{state}")

    if verbose:
        print(f"\nFinal dice: {state}, score: {game.score}")

    return game.score


def play_agent(agent_class, game, seed):
    np.random.seed(seed)
    return play_game_with_agent(agent_class(game), game)


def comparisons(iterations=1):
    agent_names = ["OneStepLookAheadAgent",
                   "RiskyAgent",
                   "CautiousAgent",
                   "MarkovDecisionProcessAgent"]
    total_agent_scores = [0, 0, 0, 0]
    for _ in range(iterations):
        seed = np.random.randint(100000)
        game = DiceGame()
        agent_scores = [play_agent(OneStepLookAheadAgent, game, seed),
                        play_agent(RiskyAgent, game, seed),
                        play_agent(CautiousAgent, game, seed),
                        play_agent(MarkovDecisionProcessAgent, game, seed)]
        total_agent_scores = [agent_scores[idx] + total_agent_scores[idx] for idx in range(len(agent_scores))]
        print(f"OneStepLookAheadAgent: {agent_scores[0]}")
        print(f"RiskyAgent: {agent_scores[1]}")
        print(f"CautiousAgent: {agent_scores[2]}")
        print(f"MarkovDecisionProcessAgent: {agent_scores[3]}")
        winners = [idx for idx in range(len(agent_scores)) if agent_scores[idx] == max(agent_scores)]
        if len(winners) > 1:
            print("Draw between")
            for winner_idx in winners:
                print(f"\t{agent_names[winner_idx]}")
        else:
            winner_idx = agent_scores.index(max(agent_scores))
            print(f"Winner is {agent_names[winner_idx]} with a score of {agent_scores[winner_idx]}")
        print("\n-----------------------\n")
    average_agent_scores = [score / iterations for score in total_agent_scores]
    print(average_agent_scores)
    best_agent_indx = average_agent_scores.index(max(average_agent_scores))
    print(f"Best agent is {agent_names[best_agent_indx]} with an average score of {max(average_agent_scores)}")


if __name__ == "__main__":
    comparisons(100)
