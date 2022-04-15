from typing import Tuple

import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import pandas as pd
import itertools
from argparse import ArgumentParser, Namespace


def read_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(data_path)
    years = df['年份'].to_numpy()
    y = df['高中阶段教育毛入学率'].to_numpy()
    raw: np.ndarray = df.drop('年份', axis=1).to_numpy()
    return years, raw, y


def generate_features(raw: np.ndarray, k: int) -> np.ndarray:
    m, n = raw.shape
    features = np.zeros((m, 3 * (n * k) ** 2))
    for i in range(k, m):
        feature = features[i]
        # Direct copy raw data from the sliding window.
        for j, (row, col) in enumerate(itertools.product(range(i - k, i), range(n))):
            feature[j] = raw[row, col]

        for j, (num, den) in zip(range(k * n, (k * n) ** 2),
                                 itertools.permutations(feature[:k * n], 2)):
            feature[j] = num / den
        feature[(k * n) ** 2: (k * n) ** 2 * 2] = feature[:(k * n) ** 2] ** 2
        feature[(k * n) ** 2 * 2: (k * n) ** 2 * 3] = np.log(feature[:(k * n) ** 2])
    return features


def pearson_selection(features: np.ndarray, y: np.ndarray, k: int, r_min: int) -> np.ndarray:
    n = features.shape[1]
    r = np.zeros(n)
    for j in range(n):
        r[j] = stats.pearsonr(features[k:, j], y[k:])[0]
    correlated_features = features[:, np.abs(r) > r_min]
    return correlated_features


def cross_validate(features: np.ndarray, y: np.ndarray, k: int):
    m, n = features.shape
    for test in range(k, m):
        reg = LinearRegression()
        reg.fit(np.delete(features, test, axis=0), np.delete(y, test, axis=0))
        # print(reg.score(features[k:], y[k:]))
        print(reg.predict([features[test]]), y[test])


def main(args: Namespace):
    years, raw, y = read_data(args.d)
    generated_features = generate_features(raw, args.window_size)
    selected_features = pearson_selection(generated_features, y, args.window_size, args.min_pearson)
    cross_validate(selected_features, y, args.window_size)


if __name__ == '__main__':
    parser = ArgumentParser(description='Predict GER (Gross Enrollment Rate) '
                                        'using linear regression.')
    parser.add_argument('-d', metavar='DATA', type=str, default='data.csv',
                        help='The path to the raw data (default: data.csv).')
    parser.add_argument('--window-size', metavar='K', type=int, default=3,
                        help='Size of the time sliding window K (default: 3).')
    parser.add_argument('--min-pearson', metavar='r', type=float, default=0.99,
                        help='Minimum Pearson correlation coefficient between feature and GER.')
    parser.add_argument('--feature-size', metavar='F', type=int, default=20,
                        help='Size of generated features (default: 20).')
    # parser.add_argument('--')
    main(parser.parse_args())
