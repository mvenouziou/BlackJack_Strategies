# probability strategy

# known: your hand: score / number of aces / specific cards
# known: dealer's hand: score / number of aces / specific cards

# given your hand and remaining cards, what is the probability of getting higher score, not bust on 1 hit, 2 hits, etc
# same for dealer except probability = 0 for x<=16 by required dealer strategy 'hit leq 16'

# example

import card_games
import numpy as np
import itertools


def prob_hit_better_than_stay_player(game, max_allowed_strat):
    diff = prob_player_win_on_hit(game, max_allowed_strat) - prob_win_on_stay(game, max_allowed_strat)
    return diff


def prob_win_on_stay(game, max_allowed_strat):
    prob_win = 1 - prob_dealer_wins_within(game, 3, max_allowed_strat)
    return prob_win


def prob_player_win_on_hit(game, max_allowed_strat):

    prob_dealer_wins = 0
    cards_set = set()

    for value in range(game.possible_scores(game.player)[0], game.possible_scores(game.player)[0] + 11):
        _, card_list = prob_x_eq(game, game.player, value, 1)
        cards_set = cards_set.union(card_list)

    for card in cards_set:
        if card in game.deck.show_cards():
            # compute probability and return card
            p = card_prob(game.deck, card)
            game.player.get_card(card)
            prob_dealer_wins += p * prob_dealer_wins_within(game, 3, max_allowed_strat)
            game.player.return_card_to_deck(card)

    prob_player_win = 1 - prob_dealer_wins

    return prob_player_win


def prob_dealer_wins_within(game, num_turns, max_allowed_strat=21):

    if game.score(game.player) > 21:  # player already busted
        prob = 1
    elif game.score(game.dealer) > 21:  # dealer already busted
        prob = 0
    elif game.score(game.dealer) >= game.score(game.player):  # dealer already met winning condition
        prob = 1
    elif game.score(game.dealer) == max_allowed_strat and game.score(game.player) > game.score(game.dealer):
        prob = 0  # for dealer strategies requiring them to 'hold' after reaching a certain threshhold
    elif num_turns <= 0:
        prob = 0

    else:  # num_turns >= 1:
        max_allowed = 21
        max_allowed_strat = 16
        min_needed = game.score(game.player)  # score dealer must match or beat

        # initialize for loop
        prob = 0
        for n in range(1, num_turns + 1):

            if n == 1:
                card_choices = set()
                for value in range(min_needed, max_allowed + 1):
                    _, cards_set = prob_x_eq(game, game.dealer, value, 1)
                    card_choices = card_choices.union(cards_set)

                for this_tuple in card_choices:
                    prob += card_prob(game.deck, this_tuple)

            else:  # this_tuple has length >= 2
                # fail on first n-1 cards
                card_choices = set()
                new_max = min(min_needed, max_allowed_strat - 1)  # dealer strategy requires 'hold' at
                # max_allowed_strat. If dealer exceeds that without beating player score, dealer loses.

                for value in range(game.possible_scores(game.dealer)[0], new_max):
                    _, card_list = prob_x_eq(game, game.dealer, value, n-1)
                    card_choices = card_choices.union(card_list)

                for cards in card_choices:
                    p1 = 1
                    p2 = 0
                    cards_taken_index = list()

                    if n == 2:
                        cards = (cards,)

                    for i in range(len(cards)):
                        p1 *= card_prob(game.deck, cards[i])
                        if p1 != 0:
                            game.dealer.get_card(cards[i])
                            cards_taken_index.append(i)

                    # succeed on last card
                    for score in range(min_needed, 22):
                        proba, _ = prob_x_eq(game, game.dealer, score, 1)
                        p2 = proba
                    prob += p1 * p2

                    # reset deck
                    for index in cards_taken_index:
                        game.dealer.return_card_to_deck(cards[index])

    return prob


def prob_x_eq(game, who, value, num_turns):
    """
    computes the probability of adding n = num_turns of cards from the deck to hand, resulting in
    hand's total value being exactly x.

    Parameters
    game: current game state from class BlackJack()
    who: perspective - game hand under consideration (game.player of game.dealer)
    value: goal value for x
    num_turns: must obtain card within this number of turns

    :return: probability of success
    """

    deck_cards = set(game.deck.show_cards())
    prob = 0

    # identify amount needed to attain value in num_turn
    additional_value_needed = [value] - np.array(game.possible_scores(who))  # second term is current hand value.
    # if hand has 'Aces' this list has length > 1

    # generate list of cards that would attain value
    if num_turns <= 1:
        next_card_needed = list()

        # add valid cards to list
        for value in additional_value_needed:
            for card in deck_cards:
                if value in game.card_value(card):
                    next_card_needed += [card]

    elif num_turns >= 2:

        tuples = list(itertools.product(deck_cards, repeat=num_turns))  # cross product of sets,
        # gives all possible combination of cards. Not all will yield desired value

        # converts tuple of cards into list of corresponding values
        next_card_needed = list()
        for value in additional_value_needed:
            for this_tuple in tuples:
                temp = [0] * num_turns  # initialize list list

                for i in range(num_turns):
                    temp[i] = game.card_value(this_tuple[i])

                    if len(temp[i]) == 1:  # card is not 'Ace'
                        temp[i] += temp[i]  # extend to length 2 (matching length for 'Ace')

                # add tuple to list if it can provide required value. need to account for one or both cards being 'Ace'
                choose_card = list(itertools.product((0, 1), repeat=num_turns))  # for selecting which card value to use
                for combo in choose_card:
                    sum_value = 0
                    for j in range(len(combo)):
                        sum_value += temp[j][combo[j]]

                    # add to list if card combination provides desired value
                    if sum_value == value:
                        next_card_needed += [this_tuple]

    next_card_needed = set(next_card_needed)  # removes duplicates

    # Calculate Probabilty of getting needed card(s)
    if num_turns == 0 and value != game.score(who):
        prob = 0
    elif num_turns == 0 and value == game.score(who):
        prob = 1

    elif num_turns == 1:
        prob = 0
        for card in next_card_needed:
            prob += card_prob(game.deck, card)

    elif num_turns >= 2:
        prob = 0
        for this_tuple in next_card_needed:
            tuple_prob = 1
            card_taken_index = list()

            for i in range(num_turns):
                tuple_prob *= card_prob(game.deck, this_tuple[i])  # probability of getting cards

                if this_tuple[i] in game.deck.show_cards():  # add card to hand
                    who.get_card(this_tuple[i])
                    card_taken_index.append(i)

            # update probability
            prob += tuple_prob

            # reset deck for next pass in loop
            for i in card_taken_index:
                who.return_card_to_deck(this_tuple[i])

    return prob, next_card_needed


def card_prob(deck, card):
    prob = deck.count(card) / len(deck.show_cards())
    return prob


if __name__ == '__main__':
    # initialize
    num_decks = 1
    deck = card_games.Deck(num_decks)
    game = card_games.BlackJack(deck)
    deck_cards = game.deck.show_cards()
    who = game.dealer

    print('PLAYER CARD:', game.player.show_cards())
    print('DEALER CARD:', game.dealer.show_cards())
    print(prob_dealer_wins_within(game, 3))
    print('PLAYER CARD:', game.player.show_cards())
    print('DEALER CARD:', game.dealer.show_cards())
