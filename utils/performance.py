from utils.optimise_gamma import game_test


if __name__ == '__main__':
    import cProfile
    import pstats

    profile = cProfile.Profile()
    profile.runcall(game_test)
    ps = pstats.Stats(profile)
    ps.print_stats()


