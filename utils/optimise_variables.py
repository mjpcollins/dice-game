import time
import numpy as np
from dice_game import DiceGame
from agents import MarkovDecisionProcessAgent
from utils.play import play_game_with_agent


def game_test(gamma=0.95, theta=0.001):

    total_score = 0
    total_time = 0
    n = 10

    np.random.seed(5)
    game = DiceGame()

    start_time = time.process_time()
    test_agent = MarkovDecisionProcessAgent(game,
                                            gamma=gamma,
                                            theta=theta)
    total_time += time.process_time() - start_time

    for i in range(n):
        start_time = time.process_time()
        score = play_game_with_agent(test_agent, game)
        total_time += time.process_time() - start_time
        total_score += score

    avg_score = total_score/n
    avg_time = total_time/n
    return avg_score, avg_time


def find_best_gamma(theta):
    best_score = 0
    best_gamma = 0
    step = 50
    # best gamma is any value between 0.945 and 0.965
    for gamma_times_1000 in range(0, 1001, step):
        gamma = gamma_times_1000 / 1000
        score, time_of_run = game_test(gamma, theta)
        print(f"Gamma: {gamma}, score: {score}, time: {time_of_run}")
        if score > best_score:
            print(f"New best score {score} with gamma {gamma}")
            best_score = score
            best_gamma = gamma
    print(f"Best gamma: {best_gamma}")
    print(f"Average score: {best_score}")


def find_best_theta(gamma):
    best_score = 0
    best_theta = 0
    step = 1

    for theta_divider in range(1, 10, step):
        theta = 0.1 / 10 ** theta_divider
        score, time_of_run = game_test(gamma, theta)
        print(f"Theta: {theta}, score: {score}, time: {time_of_run}")
        if score > best_score:
            print(f"New best score {score} with theta {theta}")
            best_score = score
            best_theta = theta
    print(f"Best theta: {best_theta}")
    print(f"Average score: {best_score}")


if __name__ == '__main__':
    find_best_theta(gamma=0.95)
