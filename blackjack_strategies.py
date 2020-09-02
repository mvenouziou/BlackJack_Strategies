# NOTE: variables defined as in module card_games.py
# game_deck = card_games.Deck()
# game = card_games.BlackJack(game_deck)

import random
import pickle
import data_collection
import counting_cards


def strategy_lists():
    strategies = {'manual': manual_strategy,
                  'probabilistic': probabilistic,
                  'ml_model': ml_model,
                  'kaggle': kaggle,
                  'hit_leq_x': hit_leq_x,
                  'random': random_choice,
                  'always_hit': always_hit,
                  'always_hold': always_hold,
                  }
    return strategies


def probabilistic(game, perspective, verbose, params):
    """
    Note: currently coded only from player's perspective.
    Results: when dealer uses hit_leq 16 and params = 16, player has 48 - 49% win rate.
    (Obtained 48.86% on 5,000 sample simulation). Without params = 16 (i.e. player doesn't know dealer's strategy)
    performance drops to

    Runtime: significantly slower than other models at approx 10.2 games / sec.
    This prevents large scale sampling of results

    Parameter:
    params: 'x' in dealer's 'hit_leq_x' strategy. Leave blank or use x=21 if dealer not using that strategy
    :return: 'hit' or 'hold' strategy
    """

    gamer = game.set_perspective(perspective)

    if perspective == 'Player':
        next_move = 'hit'
        while next_move == 'hit' and game.winner() is None:
            if verbose is True:
                game.show_hands()
                print()

            # decision criteria for hit or hold
            if counting_cards.prob_hit_better_than_stay_player(game, params) >= 0:
                next_move = 'hit'

                if verbose is True:
                    print(perspective.capitalize() + " hits...")

                gamer.draw_cards(1)
                if game.winner() is not None and verbose is True:
                    game.show_hands()
                    print()
            else:
                next_move = 'hold'

    else:  # perspective == game.dealer:

        next_move = 'hit'
        while next_move == 'hit' and game.winner() is None:
            if verbose is True:
                game.show_hands()
                print()

            # decision criteria for hit or hold
            if counting_cards.prob_dealer_wins_within(game, num_turns=3, max_allowed_strat=params) > \
                    counting_cards.prob_dealer_wins_within(game, num_turns=0, max_allowed_strat=params):
                next_move = 'hit'

                if verbose is True:
                    print(perspective.capitalize() + " hits...")
                gamer.draw_cards(1)
                if game.winner() is not None and verbose is True:
                    game.show_hands()
                    print()
            else:
                next_move = 'hold'

    game.holds(gamer, True)


def ml_model(game, perspective, verbose, params):
    # params is the model name
    # choices are
    # MLPClassifier
    # KNeighborsClassifier # SGDClassifier # tree.DecisionTreeClassifier
    # svm.SVC  # MLPClassifier
    model_name = str(params)

    # load models
    file = open('Kaggle/blackjack/models/' + model_name + '_model.pkl', 'rb')
    models_fitted = pickle.load(file)

    model_win = models_fitted['our_model_win']
    model_lose = models_fitted['our_model_lose']

    # decide between 'hit' and 'stay'
    gamer = game.set_perspective(perspective)
    next_move = 'hit'  # temp state to initiate loop. has no other effect
    turn_num = 0
    while next_move == 'hit' and game.winner() is None:
        turn_num += 1
        if verbose is True:
            game.show_hands()
            print()

        # gather data for model
        game_state = game.record_game()
        _, x, _, _ = \
            data_collection.data_shaper(game_state, for_model_creation=False)

        # default move
        next_move = 'hold'

        # decision criteria: switch to 'hit'
        if model_name in 'MLPClassifier':
            if turn_num == 1 \
                    and model_win.predict_proba(x)[0][1] > .335:
                next_move = 'hit'

            elif turn_num > 1 \
                    and model_win.predict_proba(x)[0][1] > .32:
                next_move = 'hit'
                # print('model_win predict win:', model_win.predict_proba(x)[0][1])  # win on 'hit'

        if model_name in ('MLPClassifier', 'GaussianNB', 'KNeighborsClassifier', 'DecisionTreeClassifier') \
                and model_win.predict_proba(x)[0][1] > .37:  # * model_lose.predict_proba(x)[0][0] >.25:
            next_move = 'hit'
            # print('model_win predict win:', model_win.predict_proba(x)[0][1])  # win on 'hit'
            # print('model_lose predict lose:', model_lose.predict_proba(x)[0][0])  # lose on 'stay'

        elif model_name in 'SGDClassifier':
            if turn_num == 1 \
               and model_win.decision_function(x)[0] * model_lose.decision_function(x)[0] < .08:
                next_move = 'hit'

            elif turn_num > 1 \
                and model_win.decision_function(x)[0] * model_lose.decision_function(x)[0] < 32:
                next_move = 'hit'

        # print('result:', next_move)
        # implement decision
        if next_move == 'hit':
            gamer.draw_cards(1)

            if verbose is True:
                print(perspective.capitalize() + " hits...")

                if game.winner() is not None:
                    game.show_hands()
                    print()
        else:
            if verbose is True:
                print(perspective.capitalize() + " holds...")

        if game.winner() is 'Dealer':
            print('mult:', model_win.predict_proba(x)[0] * model_lose.predict_proba(x)[0])

        file.close()
    game.holds(gamer, True)


def kaggle(game, perspective, verbose, params):
    # params is the max score where gamer 'hits'

    params = None  # not used

    gamer = game.set_perspective(perspective)
    if perspective == 'Player':
        opponent = game.dealer
    else:
        opponent = game.player
    next_move = 'hit'  # temp state to initiate loop. has no other effect

    while next_move == 'hit' and game.winner() is None:
        if verbose is True:
            game.show_hands()
            print()

        # decision criteria for hit or hold
        if game.score(gamer) < 12:
            next_move = 'hit'
        elif game.player.count("Ace") > 1 and game.score(gamer) < 18:
            next_move = 'hit'
        elif game.score(opponent) > 6 and game.score(gamer) < 17:
            next_move = 'hit'
        elif game.score(opponent) < 4 and game.score(gamer) < 13:
            next_move = 'hit'
        else:
            next_move = 'hold'

        if next_move == 'hit':
            if verbose is True:
                print(perspective.capitalize() + " hits...")
            gamer.draw_cards(1)
            if game.winner() is not None and verbose is True:
                game.show_hands()
                print()

    game.holds(gamer, True)


def always_hold(game, perspective, verbose, params):
    # params is the max score where gamer 'hits'

    params = None  # not used

    gamer = game.set_perspective(perspective)
    next_move = 'hit'  # temp state to initiate loop. has no other effect

    while next_move == 'hit' and game.winner() is None:
        if verbose is True:
            game.show_hands()
            print()

        # decision criteria for hit or hold
        if True is False:  # always hold
            next_move = 'hit'

            if verbose is True:
                print(perspective.capitalize() + " hits...")
            gamer.draw_cards(1)
            if game.winner() is not None and verbose is True:
                game.show_hands()
                print()
        else:
            next_move = 'hold'

    game.holds(gamer, True)


def always_hit(game, perspective, verbose, params):
    # params is the max score where gamer 'hits'

    params = None  # not used

    gamer = game.set_perspective(perspective)
    next_move = 'hit'  # temp state to initiate loop. has no other effect

    while next_move == 'hit' and game.winner() is None:
        if verbose is True:
            game.show_hands()
            print()

        # decision criteria for hit or hold
        if True is True:  # always hit
            next_move = 'hit'

            if verbose is True:
                print(perspective.capitalize() + " hits...")
            gamer.draw_cards(1)
            if game.winner() is not None and verbose is True:
                game.show_hands()
                print()
        else:
            next_move = 'hold'

    game.holds(gamer, True)


def random_choice(game, perspective, verbose, params):
    # params is the max score where gamer 'hits'

    params = None  # not used

    gamer = game.set_perspective(perspective)
    next_move = 'hit'  # temp state to initiate loop. has no other effect

    while next_move == 'hit' and game.winner() is None:
        if verbose is True:
            game.show_hands()
            print()

        # decision criteria for hit or hold
        x = random.randint(0, 2)
        if x == 1:
            next_move = 'hit'

            if verbose is True:
                print(perspective.capitalize() + " hits...")
            gamer.draw_cards(1)
            if game.winner() is not None and verbose is True:
                game.show_hands()
                print()
        else:
            next_move = 'hold'

    game.holds(gamer, True)


def hit_leq_x(game, perspective, verbose, params):
    # params is the max score where gamer 'hits'

    gamer = game.set_perspective(perspective)
    next_move = 'hit'  # temp state to initiate loop. has no other effect

    while next_move == 'hit' and game.winner() is None:

        if verbose is True:
            game.show_hands()
            print()

        # decision criteria for hit or hold
        if game.score(gamer) <= params:
            next_move = 'hit'

            if verbose is True:
                print(perspective.capitalize() + " hits...")
            gamer.draw_cards(1)

            if game.winner() is not None and verbose is True:
                game.show_hands()
                print()
        else:
            next_move = 'hold'

    game.holds(gamer, True)


def manual_strategy(game, perspective, verbose, params):

    verbose = None  # not used
    params = None  # not used

    gamer = game.set_perspective(perspective)

    next_move = 'hit'  # temp state to initiate loop. has no other effect
    while next_move == 'hit' and game.winner() is None:

        game.show_hands()
        next_move = input("'hit' or 'hold'?: ")
        print()

        if next_move == 'hit':
            gamer.draw_cards(1)

    game.holds(gamer, True)
