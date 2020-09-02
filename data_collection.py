import blackjack
import numpy as np
import time
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


NUM_ITERATIONS_MODEL = 200
SAMPLE_SIZE = 1000
NUM_ITERATIONS_TEST = 5000
PLAYER_DATA_COLLECT_STRATEGY = 'random'
DEALER_DATA_COLLECT_STRATEGY = 'hit_leq_x'
DEALER_STRATEGY = 'hit_leq_x'
TEST_STRATEGY = 'probabilistic'
STRATEGY_PARAMS = (16, 16)
NUM_DECKS = 1
COLLECT_DATA = False  # caution!: if 'true' this will overwrite existing data
VERBOSE_GAME_RECORDS = True
MODEL_NAME = 'MLPClassifier'
MODEL_PARAMS = ''
TRAIN_MODEL = False
APPLY_MODEL = True


def load_models_dict():
    models_dict = {'KNeighborsClassifier': KNeighborsClassifier,
                   'SGDClassifier': SGDClassifier,
                   'DecisionTreeClassifier': DecisionTreeClassifier,
                   'MLPClassifier': MLPClassifier,
                   'GaussianNB': GaussianNB}
    return models_dict


def main(model_name=MODEL_NAME, train_model=TRAIN_MODEL, apply_model=APPLY_MODEL, collect_data=COLLECT_DATA,
         model_params=MODEL_PARAMS, strategy_params=STRATEGY_PARAMS):

    model_params = model_name
    # collect training data
    if collect_data is True:

        # collect data
        final_game_results_array_data, int_game_results_array_data, win_rate = \
            collect_results(num_iteration=SAMPLE_SIZE, create_final_game_rec=False, create_int_game_rec=True,
                            player_strategy=PLAYER_DATA_COLLECT_STRATEGY,
                            dealer_strategy=DEALER_DATA_COLLECT_STRATEGY, model_params=model_params)

        # remove duplicate rows
        final_game_results_array_data = \
            np.unique(final_game_results_array_data, axis=0)
        int_game_results_array_data = \
            np.unique(int_game_results_array_data, axis=0)

        dataset_dict = {'final_game_results_array_data': final_game_results_array_data,
                        'int_game_results_array_data': int_game_results_array_data}

        # display stats
        print('Data collection -')
        print('Win rate:', win_rate)
        print('Sample Size (Original):', SAMPLE_SIZE)
        print('Sample Size (int_game results, duplicates removed):', int_game_results_array_data.shape[0])
        print('Sample Size (full_game results, duplicates removed):', final_game_results_array_data.shape[0])

        # save data to disk
        file = open('Kaggle/blackjack/datasets/random_hit_dataset2.pkl', 'wb')
        pickle.dump(dataset_dict, file)
        file.close()

    if train_model is True:

        # select model
        model_choices = load_models_dict()
        model = model_choices[model_name]

        # load data to disk
        file = open('Kaggle/blackjack/datasets/random_hit_dataset2.pkl', 'rb')
        dataset_dict = pickle.load(file)

        # final_game_results_array_data = \
        #    dataset_dict['dataset_dict'].copy()

        int_game_results_array_data = \
            dataset_dict['int_game_results_array_data'].copy()
        file.close()

        # shape features for model
        X_train_win, X_train_lost, y_train_win, y_train_lost = \
            data_shaper(int_game_results_array_data, for_model_creation=True)
        print('sample size- X_train_win:', len(X_train_win[:, 0]))
        print('winning samples:', np.sum(y_train_win))
        print('sample size- X_train_lost:', len(X_train_lost[:, 0]))
        print('losing samples:', np.sum(y_train_lost))

        # create model
        our_model_win = fit_predictor(model, X_train_win, y_train_win)
        print("\nModel Score- our_model_win:", our_model_win.score(X_train_win, y_train_win))

        our_model_lose = fit_predictor(model, X_train_lost, y_train_lost)
        print("\nModel Score- our_model_lose:", our_model_lose.score(X_train_lost, y_train_lost))

        models_fitted = {'our_model_win': our_model_win,
                         'our_model_lose': our_model_lose}

        # save model to disk
        file = open('Kaggle/blackjack/models/' + str(model_name) + '_model.pkl', 'wb')
        pickle.dump(models_fitted, file)
        file.close()

    # play games using model
    if apply_model is True:

        print("testing strategy:")
        final_game_results_array_model, int_game_results_array_model, win_rate = \
            collect_results(num_iteration=NUM_ITERATIONS_TEST, create_final_game_rec=False, create_int_game_rec=True,
                            player_strategy=TEST_STRATEGY, dealer_strategy=DEALER_STRATEGY,
                            strategy_params=strategy_params, model_params=model_params)

        print('Model results:')
        print("Win rate:", win_rate)


def fit_predictor(model, X_train, y_train):

    print('building model')

    # set model parameters
    # default model
    if model is MLPClassifier:
        our_model = model(solver='adam', activation='identity', alpha=25.0065, learning_rate='invscaling',
                          hidden_layer_sizes=(1,),
                          warm_start=True, early_stopping=False, max_iter=NUM_ITERATIONS_MODEL)

        """
        Strategy" 'hit' when model_win.predict_proba(x)[0][1] > .37
        - 36.9% win rate on 50,000
            our_model = MLPClassifier(solver='adam', activation='identity', alpha=25.0065, learning_rate='adaptive',
                              hidden_layer_sizes=(1)
                              , warm_start=True, early_stopping=False,
                              max_iter=NUM_ITERATIONS_MODEL)
                
        - 36.6% win rateon 50,000
        our_model = MLPClassifier(solver='adam', activation='relu', alpha=29.02,learning_rate='adaptive',
                          hidden_layer_sizes=(50,100,200)
                          , warm_start=True, early_stopping=False,
                          max_iter=NUM_ITERATIONS_MODEL)
        """

    elif model is KNeighborsClassifier:
        our_model = model(weights='distance', algorithm='brute')

    elif model is GaussianNB:
        our_model = model()

    elif model is DecisionTreeClassifier:
        our_model = model()

    elif model is SGDClassifier:
        our_model = model(loss="log", penalty="elasticnet", max_iter=NUM_ITERATIONS_MODEL)

    # fit model
    our_model.fit(X_train, y_train)

    return our_model


def data_shaper(game_array, for_model_creation):

    if for_model_creation is True:
        print('beginning data_shaper...')

    randomizer = np.random.default_rng()

    # add/remove features
    feature_update_game_array = engineer_data(game_array)

    # create 'hit' and 'hold scenarios
    hit_array = strip_new(feature_update_game_array)  # revert to prior hand info
    stay_array = strip_old(feature_update_game_array)  # use current hand info

    if for_model_creation is False:  # for playing individual games (game_array has only one row)
        game_wins = stay_array
        game_lost = stay_array

    else:  # combine arrays for training model
        full_array = np.vstack((hit_array, stay_array))
        randomizer.shuffle(full_array)

        # sort by wins/losses
        sort_order = np.argsort(full_array[:, -1])
        sorted_array = full_array[sort_order, :]

        # balance wins/losses
        num_wins = np.sum(sorted_array, axis=0)[-1]

        # separate 'wins' and 'lose'
        game_wins = sorted_array[-int(1 * num_wins):, :]
        game_lost = sorted_array[:-int(1 * num_wins), :]

        # reshuffle rows
        randomizer.shuffle(game_wins)
        randomizer.shuffle(game_lost)

    # separate into features / target
    X_train_win = game_wins[:, 1: -1]  # drops 'hit/stay' and 'win/lose'
    y_train_win = game_wins[:, 0]  # indicates 'hit' or 'stay'

    X_train_lost = game_lost[:, 1: -1]
    y_train_lost = game_lost[:, 0]

    # standardize data for improved model performance
    X_train_win = standardize(X_train_win)
    X_train_lost = standardize(X_train_lost)

    return X_train_win, X_train_lost, y_train_win, y_train_lost


def standardize(X_train):
    scaler = StandardScaler().fit(X_train)
    scaler.transform(X_train)
    return X_train


def engineer_data(game_array):
    # ** IMPORTANT: Leave columns 0 - 5 and final column's values and positions unaltered **
    rows = game_array.shape[0]

    # delete unwanted info
    # game_array = np.delete(game_array,np.s_[7:-1], axis=1)

    # add new features
    column = np.ones(rows)
    game_array = np.insert(game_array, -1, column, axis=1)

    column = (1 - game_array[:, 1])*(1 - game_array[:, 5])  # interaction term
    game_array = np.insert(game_array, -1, column, axis=1)

    column = np.log(abs((1 - game_array[:, 1])*(1 - game_array[:, 5])) + .001) / np.log(2)
    game_array = np.insert(game_array, -1, column, axis=1)

    column = (game_array[:, 1])**2
    game_array = np.insert(game_array, -1, column, axis=1)

    column = (game_array[:, 5])**2
    game_array = np.insert(game_array, -1, column, axis=1)

    return game_array


def strip_old(game_array):
    new_array1 = game_array[game_array[:, 0] == 0]  # when played 'hold' on last move
    new_array1 = np.delete(new_array1, [4, 5], axis=1)

    new_array2 = game_array[game_array[:, 3] == 2]  # busted on first hit
    new_array2 = np.delete(new_array2, [4, 5], axis=1)
    new_array2[:, 0] = 1  # player 'hit' on previous game

    new_array = np.vstack((new_array1, new_array2))

    return new_array


def strip_new(game_array):
    new_array = game_array[game_array[:, 3] >= 3]
    new_array = np.delete(new_array, [1, 2], axis=1)
    new_array[:, 0] = 1  # all players 'hit' on previous game
    new_array[:, 3] -= 1  # reduce length of hand

    return new_array


def collect_results(num_iteration, create_final_game_rec, create_int_game_rec,
                    player_strategy, dealer_strategy, strategy_params=STRATEGY_PARAMS, model_params=MODEL_PARAMS):

    print('Beginning data collection.')
    # Start the stopwatch / counter
    t1_start = time.perf_counter()

    # first iteration
    new_full_game_results_array, new_int_game_results_array, wins = \
        blackjack.main(player_strategy=player_strategy, dealer_strategy=dealer_strategy,
                       num_decks=NUM_DECKS, strategy_params=strategy_params, model_params=model_params,
                       verbose=False, verbose_game_records=False, play_again_prompt=False)
    num_wins = wins
    print('win:', wins)

    # initialize arrays to store full data set
    full_game_results_array = np.zeros((num_iteration, new_full_game_results_array.shape[1]))
    int_game_results_array = np.zeros((num_iteration, new_int_game_results_array.shape[1]))

    # update arrays
    full_game_results_array[0, :] = new_full_game_results_array
    int_game_results_array[0, :] = new_int_game_results_array

    # remaining iterations
    for n in range(1, num_iteration):
        new_full_game_results_array, new_int_game_results_array, wins = \
            blackjack.main(player_strategy=player_strategy, dealer_strategy=dealer_strategy,
                           num_decks=NUM_DECKS, strategy_params=strategy_params, model_params=model_params,
                           verbose=False, verbose_game_records=False, play_again_prompt=False)

        # update wins count
        if wins == 1:
            print('WINNER!:')
        else:
            print('LOST!')
        num_wins += wins

        # update game results arrays
        if create_final_game_rec is True:
            full_game_results_array[n, :] = new_full_game_results_array

        if create_int_game_rec is True:
            int_game_results_array[n, :] = new_int_game_results_array

        if n % 100 == 1:
            # Stop the stopwatch / counter
            t1_stop = time.perf_counter()

            print("Elapsed time:", t1_stop, t1_start)
            print(n, 'samples collected......')
            print("total runtime:", t1_stop - t1_start)
            print('current win rate:', num_wins/n, "\n\n")

    win_rate = num_wins/num_iteration

    return full_game_results_array, int_game_results_array, win_rate


if __name__ == '__main__':
    main()
