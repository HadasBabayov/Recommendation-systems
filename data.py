# Hadas Babayov 322807629
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def watch_data_info(data):
    for d in data:
        # This function returns the first 5 rows for the object based on position.
        # It is useful for quickly testing if your object has the right type of data in it.
        print(d.head())

        # This method prints information about a DataFrame including the index dtype and column dtypes, non-null values and memory usage.
        print(d.info())

        # Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.
        print(d.describe(include='all').transpose())


def print_data(data):
    ratings = data[0]

    print('Answer 1:')
    num_of_rating_users = len(ratings['userId'].unique())
    num_of_rating_movies = len(ratings['movieId'].unique())
    num_of_ratings = len(ratings)
    print("Number of ratings: ", num_of_ratings)
    print("Number of unique movies: ", num_of_rating_movies)
    print("Number of unique users: ", num_of_rating_users)

    print('Answer 2:')
    max_rate_per_movie = ratings['movieId'].value_counts().max()
    min_rate_per_movie = ratings['movieId'].value_counts().min()
    print("Min: ", min_rate_per_movie)
    print("Max: ", max_rate_per_movie)

    print('Answer 3:')
    min_rate_per_user = ratings['userId'].value_counts().min()
    max_rate_per_user = ratings['userId'].value_counts().max()
    print("Min: ", min_rate_per_user)
    print("Max: ", max_rate_per_user)


def plot_data(data, plot=True):
    ratings = data[0].drop(['userId'], axis=1)
    rate_appear = ratings['rating'].value_counts().reset_index(name='rates').sort_values(by=['index'], ascending=True)
    x = rate_appear['index']
    y = rate_appear['rates']
    plt.bar(x, y, color='maroon', width=0.4)
    plt.ylabel('Number of movies that get this rate')
    plt.xlabel('Rate value')
    plt.xticks(x, x)
    if plot:
        plt.show()
