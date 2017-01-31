import pandas
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

filename = "data/current_version.csv"
print("Loading '%s'" % filename)
df = pandas.read_csv(filename)
n = len(df)
train = df[n//10:]
test = df[:n//10+1]

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
for c in MLPClassifier, KNeighborsClassifier, SVC, DecisionTreeClassifier, \
         RandomForestClassifier, AdaBoostClassifier, GaussianNB:
    clf = c()
    clf.fit(train[FEATURE_NAMES], train[["ATTITUDE"]].squeeze())
    predicted = clf.predict(test[FEATURE_NAMES])
    difference = test[["ATTITUDE"]].subtract(predicted, axis=0).astype(bool).sum(axis=0)
    score = 1 - difference[0] / test.shape[0]
    print("%s: %f" % (c.__name__, score))
