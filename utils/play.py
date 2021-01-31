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
    agent_names = ["AlwaysHoldAgent",
                   "PerfectionistAgent",
                   "OneStepLookAheadAgent",
                   "RiskyAgent",
                   "CautiousAgent",
                   "MarkovDecisionProcessAgent",
                   "MarkovDecisionProcessAgentAdjusted"]
    total_agent_scores = [0 for i in range(len(agent_names))]
    for game_round in range(iterations):
        seed = np.random.randint(1000)
        game = DiceGame()
        agent_scores = [play_agent(AlwaysHoldAgent, game, seed),
                        play_agent(PerfectionistAgent, game, seed),
                        play_agent(OneStepLookAheadAgent, game, seed),
                        play_agent(RiskyAgent, game, seed),
                        play_agent(CautiousAgent, game, seed),
                        play_agent(MarkovDecisionProcessAgent, game, seed),
                        play_agent(MarkovDecisionProcessAgentAdjusted, game, seed)]
        total_agent_scores = [agent_scores[idx] + total_agent_scores[idx] for idx in range(len(agent_scores))]
        for i in range(len(agent_names)):
            print(f"{agent_names[i]}: {agent_scores[i]}")
        winners = [idx for idx in range(len(agent_scores)) if agent_scores[idx] == max(agent_scores)]
        if len(winners) > 1:
            print("Draw between")
            for winner_idx in winners:
                print(f"\t{agent_names[winner_idx]}")
        else:
            winner_idx = agent_scores.index(max(agent_scores))
            print(f"Winner is {agent_names[winner_idx]} with a score of {agent_scores[winner_idx]}")
        print("\n-----------------------\n")
        write_results(agent_scores, agent_names, game_round)
    average_agent_scores = [score / iterations for score in total_agent_scores]
    print(average_agent_scores)
    best_agent_indx = average_agent_scores.index(max(average_agent_scores))
    print(f"Best agent is {agent_names[best_agent_indx]} with an average score of {max(average_agent_scores)}")


def write_results(results, agents, game_round):
    with open("../data/play_results.csv", "a") as F:
        for i in range(len(agents)):
            F.write(f"{game_round}, {agents[i]}, {results[i]}\n")


if __name__ == "__main__":
    comparisons(100)
