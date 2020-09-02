import random
import numpy as np


class Deck:

    def __init__(self, num_decks):
        self.__cards = (list(range(2, 11)) +
                        ['Jack', 'Queen', 'King', 'Ace']) * 4 * num_decks
        self.__num_decks = num_decks

    def remove_card(self, card=0):
        if card == 0:  # take next card in deck
            chosen_card = self.__cards.pop(0)
        else:  # specify the card
            ind = self.__cards.index(card)
            chosen_card = self.__cards.pop(ind)
        return chosen_card

    def add_card(self, card):
        self.__cards.append(card)

    def shuffle(self):
        random.shuffle(self.__cards)
        return self

    def show_cards(self):
        return self.__cards

    def set_cards(self, card_list):
        self.__cards = card_list

    def get_num_decks(self):
        return self.__num_decks

    def count(self, card):
        num_occurrences = self.__cards.count(card)
        return num_occurrences


class Hand(Deck):

    def __init__(self, deck):
        num_decks = deck.get_num_decks()

        Deck.__init__(self, num_decks)
        self.set_cards([])
        self.deck = deck

    def draw_cards(self, number_of_cards=1):
        while number_of_cards > 0:
            card = self.deck.remove_card()
            card_list = self.show_cards() + [card]
            self.set_cards(card_list)
            number_of_cards -= 1

    def get_card(self, card):
        card = self.deck.remove_card(card)
        card_list = self.show_cards() + [card]
        self.set_cards(card_list)

    def return_card_to_deck(self, card):
        self.remove_card(card)
        self.deck.add_card(card)


class BlackJack(Hand):

    def __init__(self, deck):
        Hand.__init__(self, deck)
        deck.shuffle()

        self.dealer = Hand(deck)
        self.dealer.draw_cards(1)
        self.__dealer_hold = False

        self.player = Hand(deck)
        self.player.draw_cards(2)
        self.__player_hold = False

        self.deck = deck

    def show_hands(self):
        print("Player's hand:", self.player.show_cards())
        print("Dealer's hand:", self.dealer.show_cards())

    def holds(self, who, is_true):
        if who == self.dealer:
            self.__dealer_hold = is_true
        if who == self.player:
            self.__player_hold = is_true

    def card_value(self, card):
        if card == 'Ace':
            values = [1, 11]
        elif card in ['Jack', 'Queen', 'King']:
            values = [10]
        else:
            values = [card]
        return values

    def possible_scores(self, who):
        num_aces = who.count('Ace')
        temp_list = list(map(lambda y: self.card_value(y)[0], who.show_cards()))
        possible_scores_list = [sum(temp_list)]

        while num_aces > 0:
            next_score = possible_scores_list[-1] + 10  # Ace previously counted as 1
            if next_score <= 21:
                possible_scores_list.append(next_score)
            num_aces -= 1

        return possible_scores_list

    def score(self, who):
        scores_list = self.possible_scores(who)
        highest_score = scores_list[-1]
        return highest_score

    def winner(self):
        has_won = None

        player_score = self.score(self.player)
        dealer_score = self.score(self.dealer)

        if player_score > 21:  # player busts
            has_won = 'Dealer'
        elif dealer_score > 21:  # dealer busts
            has_won = 'Player'
        elif dealer_score >= player_score and self.__player_hold is True:
            has_won = 'Dealer'
        elif dealer_score < player_score and self.__dealer_hold is True:
            has_won = 'Player'

        return has_won

    def set_perspective(self, perspective):
        if perspective == 'Player':
            gamer = self.player
        else:  # perspective == 'Dealer':
            gamer = self.dealer
        return gamer

    def record_game(self):
        num_decks = self.deck.get_num_decks()

        card_options = list(range(2, 11)) + ['Jack', 'Queen', 'King', 'Ace']
        # gathers and normalizes game state data
        records = list()
        # player current card info
        records.append(self.possible_scores(self.player)[0] > 21)  # True iff player busted iff last move was 'hit'
        records.append(self.possible_scores(self.player)[0])  # percent of max score (21) with aces=1
        records.append(self.player.count("Ace") / (4 * num_decks))  # percent of aces held by player
        records.append(len(self.player.show_cards()))
        # player card info ignoring last card
        temp_list = self.player.show_cards()[:-1]  # list of cards prior to final card
        self.temp_player = Hand(Deck(num_decks))  # temp player from new deck
        self.temp_player.set_cards(temp_list)  # assign hand to temp player
        records.append(self.possible_scores(self.temp_player)[0] / 21)
        records.append(self.temp_player.count("Ace") / (4 * num_decks))
        # dealer card info
        records.append(self.possible_scores(self.dealer)[0] / 21)
        records.append(self.dealer.count("Ace") / (4 * num_decks))
        records.append(len(self.dealer.show_cards()))
        for i in range(len(card_options)):
            records.append(self.deck.count(card_options[i]) / (4 * num_decks))  # percent remaining in deck
        records.append(self.winner() == 'Player')  # winner recorded in last column

        records = np.array(records).reshape(1, -1)

        return records


# For testing
if __name__ == '__main__':
    print("")
