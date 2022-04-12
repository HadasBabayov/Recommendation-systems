# Hadas Babayov 322807629
import sys
import pandas as pd
import numpy as np
import heapq
from sklearn.metrics.pairwise import pairwise_distances


class collaborative_filtering:
    def __init__(self):
        self.user_based_matrix = []
        self.item_based_metrix = []

    def get_matrix(self, data):
        ratings = data[0]
        movies_col = ratings['movieId']
        users_col = ratings['userId']

        unique_users = np.sort(users_col.drop_duplicates().to_numpy()).tolist()
        unique_movies = np.sort(movies_col.drop_duplicates().to_numpy()).tolist()

        # Dictionary for users and movies.
        self.users_map = {key: val for val, key in enumerate(unique_users)}
        self.movies_map = {key: val for val, key in enumerate(unique_movies)}
        self.second_movies_map = {key: val for key, val in enumerate(unique_movies)}

        # Rate matrix.
        matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
        self.matrix = matrix.to_numpy()
        df = pd.DataFrame(self.matrix)
        df_np = df.to_numpy()
        mean = df.mean(axis=1).to_numpy().reshape(-1, 1)
        mean.round(2)
        diff = df_np - mean
        diff[np.isnan(diff)] = 0
        diff.round(2)
        return diff,mean



    def create_fake_user(self, rating):
        user_id = 283238

        movies_subset = self.data[1]
        genres_col = movies_subset['genres']
        movie_id_col = movies_subset['movieId']

        # Rate genres.
        d = {
            'Romance': 0.5,
            'Comedy': 4,
            'Action': 5,
            'Drama': 0.5,
            'IMAX': 1,
            'War': 1,
            'Adventure': 2.5,
            'Fantasy': 1,
            'Mystery': 0.5,
            'Crime': 1.5,
            'Film-Noir': 2,
            'Thriller': 2,
            'Horror': 1,
            'Documentary': 3,
            'Children': 0.5,
            'Western': 2,
            'Sci-Fi': 4,
            'Musical': 1.5,
        }

        # Calculate movies ratings by genres dictionary.
        rates_of_movies = []
        for i in range(len(movies_subset)):
            movie_rate = 0
            movie_genre = genres_col[i]
            # Average.
            num_of_genres = movie_genre.count('|') + 1
            for key in d.keys():
                if key in movie_genre:
                    movie_rate += d[key]
            movie_rate /= num_of_genres
            rates_of_movies.append(movie_rate)

        for i in range(50):
            id_of_movie = movie_id_col[i]
            rate = rates_of_movies[i]
            new = {'userId': user_id, 'movieId': id_of_movie, 'rating': rate}
            # Add row
            rating = rating.append(new, ignore_index=True)

        return rating

    def create_user_based_matrix(self, data):
        self.data = data
        ratings = data[0]
        # For adding fake user.
        ratings = self.create_fake_user(ratings)
        data = (ratings, data[1])
        self.new_data = data
        diff, mean = self.get_matrix(data)
        # Calculate user x user similarity matrix.
        user_similarity = 1 - pairwise_distances(diff, metric='cosine')
        self.user_similarity = pd.DataFrame(user_similarity).round(2).to_numpy()
        pd.DataFrame(user_similarity).round(2)
        pd.DataFrame(user_similarity.dot(diff).round(2))
        # Prediction matrix.
        pred = mean + user_similarity.dot(diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T
        pred.round(2)
        self.user_based_matrix = pred

    def create_item_based_matrix(self, data):
        self.data = data
        # for adding fake user
        diff, mean = self.get_matrix(data)

        # calculate user x item similarity matrix
        item_similarity = 1 - pairwise_distances(diff.T, metric='cosine')
        self.item_similarity = pd.DataFrame(item_similarity).round(2).to_numpy()
        pd.DataFrame(item_similarity).round(2)

        # Prediction matrix.
        pred = (mean + diff.dot(item_similarity) / np.array(
            [np.abs(item_similarity).sum(axis=1)]))
        pred.round(2)
        self.item_based_metrix = pred

    def predict_movies(self, user_id, k, is_user_based=True):
        if is_user_based:
            pred = self.user_based_matrix
        else:
            pred = self.item_based_metrix

        new_user_id = self.users_map[int(user_id)]

        one_row_pred = pred[new_user_id]
        row_with_index = []
        for i, val in enumerate(one_row_pred):
            row_with_index.append((i, val))

        one_row_matrix = self.matrix[new_user_id]
        # Save nan indexes.
        index_of_nan = np.argwhere(np.isnan(one_row_matrix))

        new_row_pred = []
        for i in index_of_nan:
            new_row_pred.append(row_with_index[i[0]])

        # Sort by rate value.
        new_row_pred.sort(key=lambda tuple: tuple[1])
        k_largest = new_row_pred[-k:]

        movies_id = []
        for pair in k_largest:
            id = pair[0]
            movies_id.append(self.second_movies_map[id])

        movies_subset = self.data[1]
        # Map from movie id to the title.
        d = {}
        for id, title in zip(movies_subset['movieId'], movies_subset['title']):
            d[id] = title

        titles = []
        for i in movies_id:
            titles.append(d[i])
        titles.reverse()

        # Return the titles of the k prediction movies.
        return titles

    def predict_movies_ret_ids(self, user_id, k, is_user_based):
        if is_user_based:
            pred = self.user_based_matrix
        else:
            pred = self.item_based_metrix

        new_user_id = self.users_map[int(user_id)]

        one_row_pred = pred[new_user_id]
        row_with_index = []
        for i, val in enumerate(one_row_pred):
            row_with_index.append((i, val))

        one_row_matrix = self.matrix[new_user_id]
        index_of_nan = np.argwhere(np.isnan(one_row_matrix))

        new_row_pred = []
        for i in index_of_nan:
            new_row_pred.append(row_with_index[i[0]])

        new_row_pred.sort(key=lambda tuple: tuple[1])
        k_largest = new_row_pred[-k:]

        movies_id = []
        for pair in k_largest:
            id = pair[0]
            movies_id.append(self.second_movies_map[id])

        return movies_id