# Hadas Babayov 322807629
import math

from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
# Import Pandas
import pandas as pd


def precision_10(test_set, cf, is_user_based=True):
    test_set = test_set[(test_set.rating == 4.0) | (test_set.rating == 5.0) | (test_set.rating == 4.5)]
    user_to_movies = test_set.groupby('userId')['movieId'].apply(list).reset_index(name='list').values

    k = 10
    hit_per_user = []
    for (u, m) in user_to_movies:
        hits = 0
        # Predict 10 movies per this user.
        movies_id = cf.predict_movies_ret_ids(str(u), k, is_user_based)
        # Count hits.
        for id in movies_id:
            if id in m:
                hits += 1
        hit_per_user.append(hits / k)

    # Calculate val.
    val = sum(hit_per_user) / len(hit_per_user)
    print("Precision_k: " + str(val))


def ARHA(test_set, cf, is_user_based=True):
    test_set = test_set[(test_set.rating == 4.0) | (test_set.rating == 5.0) | (test_set.rating == 4.5)]
    user_to_movies = test_set.groupby('userId')['movieId'].apply(list).reset_index(name='list').values

    k = 10
    hit_per_user = []
    for (u, m) in user_to_movies:
        hits = 0
        # Predict 10 movies per this user.
        movies_id = cf.predict_movies_ret_ids(u, k, is_user_based)
        movies_id.reverse()
        # Count hits.
        for index, id in enumerate(movies_id):
            if id in m:
                hits += 1 / (index + 1)
        hit_per_user.append(hits)

    # Calculate val.
    val = sum(hit_per_user) / len(hit_per_user)
    print("ARHR: " + str(val))


def RSME(test_set, cf, is_user_based=True):
    if is_user_based:
        pred_matrix = cf.user_based_matrix
    else:
        pred_matrix = cf.item_based_metrix

    # Calculate val.
    users = test_set['userId']
    movies = test_set['movieId']
    pred_rates = np.array([pred_matrix[cf.users_map[u], cf.movies_map[m]] for u, m in zip(users, movies)])
    val = mean_squared_error(test_set['rating'].to_numpy(), pred_rates, squared=False)

    print("RMSE: " + str(val))
