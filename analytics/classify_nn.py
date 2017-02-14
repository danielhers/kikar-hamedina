import dynet as dy
import numpy as np
import pandas

filename = "data/current_version.csv"
print("Loading '%s'" % filename)
df = pandas.read_csv(filename)

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

y_vals = sorted(list(set(map(int, df[["ATTITUDE"]].get_values()))))
y_size = max(y_vals) + 1

LAYERS = 2
LAYER_DIM = 50
DROPOUT = .5
ITERATIONS = 100

m = dy.Model()
sgd = dy.SimpleSGDTrainer(m)

pW = []
pb = []
for i in range(LAYERS):
    in_dim = len(FEATURE_NAMES) if i == 0 else LAYER_DIM
    out_dim = LAYER_DIM if i < LAYERS else y_size
    pW.append(m.add_parameters((out_dim, in_dim)))
    pb.append(m.add_parameters(out_dim))


def evaluate(features):
    x = dy.inputVector(features)
    for i in range(LAYERS):
        W = dy.parameter(pW[i])
        b = dy.parameter(pb[i])
        x = dy.dropout(x, DROPOUT)
        x = dy.tanh(W * x + b)
    return x


for iter in range(ITERATIONS):
    print("iteration %d" % (iter + 1))
    # np.random.shuffle(train)
    for i, sample in enumerate(train[FEATURE_NAMES + ["ATTITUDE"]].get_values()):
        out = evaluate(sample[:-1])
        loss = dy.pickneglogsoftmax(out, int(sample[-1]))
        loss.value()
        loss.backward()
        sgd.update()
    sgd.update_epoch()

probs = []
for sample in test[FEATURE_NAMES].get_values():
    out = evaluate(sample)
    y = dy.softmax(out).value()
    probs.append(y)

y_pred = np.argmax(probs, axis=1)
y = list(map(int, test[["ATTITUDE"]].get_values()))
acc = np.mean(y_pred == y)
print("accuracy = %.3f" % acc)
