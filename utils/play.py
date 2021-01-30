

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


if __name__ == "__main__":
    from agents import *
    from dice_game import DiceGame
    import numpy as np

    seed = int(sum(np.random.random(100)) * 100)
    game = DiceGame()

    np.random.seed(seed)
    agent1 = OneStepLookAheadAgent(game)
    play_game_with_agent(agent1, game, verbose=True)

    print("\n")

    np.random.seed(seed)
    agent2 = RiskyAgent(game)
    play_game_with_agent(agent2, game, verbose=True)

    print("\n")

    np.random.seed(seed)
    agent2 = CautiousAgent(game)
    play_game_with_agent(agent2, game, verbose=True)


