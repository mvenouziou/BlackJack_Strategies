import blackjack_strategies as strategy
import card_games

PLAYER_STRATEGY = 'probabilistic'
DEALER_STRATEGY = 'hit_leq_x'
STRATEGY_PARAMS = (16, 16)  # controls 'hit_leq_x' and 'probabalistic' strategies
MODEL_PARAMS = 'MLPClassifier'
NUM_DECKS = 1
VERBOSE = True
VERBOSE_GAME_RECORDS = False
PLAY_AGAIN_PROMPT = True
CREATE_GAME_REC = False
CREATE_INT_GAME_REC = False


def main(player_strategy=PLAYER_STRATEGY, dealer_strategy=DEALER_STRATEGY, num_decks=NUM_DECKS,
         strategy_params=STRATEGY_PARAMS, model_params=MODEL_PARAMS, verbose=VERBOSE,
         verbose_game_records=VERBOSE_GAME_RECORDS, play_again_prompt=PLAY_AGAIN_PROMPT, num_wins=0, games_played=0):

    # Initialize Game
    strategies = strategy.strategy_lists()  # load player strategies dictionary
    game_deck = card_games.Deck(num_decks)
    game = card_games.BlackJack(game_deck)

    # Player's Turn  ####
    if verbose is True:
        print("Player's Turn\n" + "-------------")
    # Implement Strategy
    if player_strategy == 'ml_model':
        strategies[player_strategy](game, perspective='Player', verbose=verbose, params=model_params)
    elif player_strategy == 'probabilistic' and dealer_strategy== 'hit_leq_x':
        strategies[player_strategy](game, perspective='Player', verbose=verbose, params=strategy_params[1])
    else:
        strategies[player_strategy](game, perspective='Player', verbose=verbose, params=strategy_params[0])

    # End Turn
    if verbose is True:
        if game.score(game.player) > 21:
            print("Player busts!")
        else:
            print("Player holds.")

    # Get intermediate Game State  ####
    intermediate_game_record = game.record_game()
    if verbose_game_records is True:
        print("Game state at end of player's turn:\n", intermediate_game_record, "\n")

    # Dealer's Turn  ####
    if game.winner() is None:
        if verbose is True:
            print("Dealer's Turn\n" + "-------------")
        # Implement Strategy
        strategies[dealer_strategy](game, perspective='Dealer', verbose=verbose, params=strategy_params[1])
        # End Turn
        if verbose is True:
            if game.score(game.dealer) > 21:
                print("Dealer busts!")
            else:
                print("Dealer holds.")

    # Declare Winner  ####
    if verbose is True:
        game.show_hands()
        print("\n" + game.winner(), "Wins!")

    # Create Game Record  ####
    game_record = game.record_game()
    games_played += 1
    if verbose_game_records is True:
        print("Game records:\n", game_record, "\n")
    if game.winner() is 'Player':
        num_wins += 1
    win_rate = num_wins / games_played

    # Update Intermediate Game State to include eventual winner
    if game.winner() == 'Player':
        intermediate_game_record[0,-1] = 1

    # Play Again?  ####
    if play_again_prompt is True:
        play_again = input("Play again? (y/n): ")
    else:
        play_again = 'n'

    if play_again == 'y':
        if verbose is True:
            print("\nDealing cards...")
        main(player_strategy, dealer_strategy, num_decks, strategy_params, model_params,
             verbose, verbose_game_records, play_again_prompt,
             num_wins, games_played)
        # note: previous game records are overridden
    else:
        if verbose is True:
            print("Thank you for playing!")
            print("Win rate:", win_rate)

    return game_record, intermediate_game_record, win_rate


if __name__ == '__main__':
    main()
