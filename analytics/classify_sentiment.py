from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy.stats import randint as sp_randint
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

filename = "data/current_version.csv"
print("Loading '%s'" % filename)
df = pandas.read_csv(filename)

hist, _ = np.histogram(df[["ATTITUDE"]].squeeze(), 100)
print("\n".join("%d: %.1f%%" % (i + 1, 100 * h / hist.sum()) for (i, h) in enumerate(hist) if h))

n = len(df)
train = df[2 * n // 10:]
test = df[n // 10 + 1:2 * n // 10:]

FEATURE_NAMES = [
    "POST_LEN_MESSAGE",
    "COMMENT_LEN_MESSAGE",
    "COMMENTATOR_LIKED_POST",
    "HAS_NAME_OF_POST_WRITER_MK_IN_COMMENT",
    # "IDS_OF_MKS_MENTIONED_IN_COMMENT",
    "NUM_OF_COMMENTS_BY_COMMENTATOR_ON_POST",
    # "COMMENTATOR_ID",
    # "POLITICAL_WING_HATNUA_LEFT",
    # "POLITICAL_WING_HATNUA_CENTER",
    "IS_COALITION",
    # "PARTY_NAME",
    "IS_FEMALE",
    "AGE",
    "MK_POLITICAL_STATUS",
    "IS_CURRENT_OR_PAST_PARTY_LEADER",
    # "IS_CURRENT_OR_PAST_PM_CANDIDATE",
    # "IS_PM",
    # "POST_PUBLICATION_TIMESTAMP",
    # "POST_PUBLICATION_DATE",
    "POST_PUBLICATION_DAYS_FROM_RESEARCH_START_DATE",
    "POST_WITH_PHOTO",
    "POST_WITH_LINK",
    "POST_WITH_VIDEO",
    "POST_WITH_STATUS",
    "POST_WITH_TEXT_ONLY",
    "POST_IN_HEBREW",
    "POST_IN_ENGLISH",
    "POST_IN_ARABIC",
    "POST_IN_OTHER",
    "DAYS_FROM_ELECTION",
    "DAYS_FROM_THREE_TEENAGER_KIDNAP",
    "DAYS_FROM_PROTECTIVE_EDGE_OFFICIAL_START_DATE",
    "DAYS_FROM_PROTECTIVE_EDGE_OFFICIAL_END_DATE",
    "DAYS_FROM_DUMA_ARSON_ATTACK",
    "DAYS_FROM_THIRD_INTIFADA_START_DATE",
    "DAYS_FROM_MK_BIRTHDAY",
    "POST_PUBLISHED_ON_SATURDAY",
    "COMMENT_PUBLISHED_ON_SATURDAY",
    "NUM_OF_COMMENTS_BY_COMMENTATOR_ID_ON_GIVEN_MK_POSTS",
    "NUM_OF_LIKES_BY_COMMENTATOR_ID_ON_GIVEN_MK_POSTS",
    "RATIO_OF_COMMENTS_BY_COMMENTATOR_ID_ON_GIVEN_MK_POSTS",
    "RATIO_OF_LIKES_BY_COMMENTATOR_ID_ON_GIVEN_MK_POSTS",
]


def classify(clf):
    clf.fit(train[FEATURE_NAMES], train[["ATTITUDE"]].squeeze())
    predicted = clf.predict(test[FEATURE_NAMES])
    difference = test[["ATTITUDE"]].subtract(predicted, axis=0).astype(bool).sum(axis=0)
    score = 1 - difference[0] / test.shape[0]
    print("%s: %f" % (clf, score))
    return score


CLASSIFIERS = MLPClassifier(), KNeighborsClassifier(), SVC(), DecisionTreeClassifier(), \
              RandomForestClassifier(bootstrap=True, max_features=26, min_samples_leaf=6,
                                     min_samples_split=7, criterion="gini", max_depth=4), \
              AdaBoostClassifier(), GaussianNB(), DummyClassifier()
scores = {c: classify(c) for c in CLASSIFIERS}
plt.bar(range(len(scores)), scores.values(), align='center')
plt.xticks(range(len(scores)), [c.__class__.__name__ for c in scores], rotation=45)
plt.show()

clf = MLPClassifier(n_estimators=20)


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# specify parameters and distributions to sample from
param_dist = {"alpha": [.1, .01, .001],
              "momentum": [.7, .9],
              "learning_rate_init": [.1, .01, .001],
              "hidden_layer_sizes": (50,),
              "max_iter": 10000,
              "batch_size": range(10, 600),
              "algorithm": ('sgd', 'adam', 'adagrad'),
              "random_state": 1,
              "activation": ("logistic", "tanh", "relu"),
              "learning_rate": "constant",
              }

# run randomized search
n_iter_search = 200
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)

start = time()
random_search.fit(train[FEATURE_NAMES], train[["ATTITUDE"]].squeeze())
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)
