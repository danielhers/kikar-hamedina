import dynet as dy
import numpy as np
import pandas

filename = "data/current_version.csv"
print("Loading '%s'..." % filename)
df = pandas.read_csv(filename)

df = df.sample(frac=1, random_state=1).reset_index(drop=True)
df = df.replace(np.nan, "", regex=True)
n = len(df)
train = df[2 * n // 10:]  # all but the first 1/5
test = df[n // 10 + 1:2 * n // 10:]  # the first 1/5 without the first 1/10 (which is kept unseen)

TEXT_FEATURES = [
    "post_content",
    "comment_content_processed",
]

NUMERIC_FEATURES = [
    "POST_LEN_MESSAGE",
    "COMMENT_LEN_MESSAGE",
    "COMMENTATOR_LIKED_POST",
    # "HAS_NAME_OF_POST_WRITER_MK_IN_COMMENT",
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
    # "POST_PUBLICATION_DAYS_FROM_RESEARCH_START_DATE",
    # "POST_WITH_PHOTO",
    # "POST_WITH_LINK",
    # "POST_WITH_VIDEO",
    # "POST_WITH_STATUS",
    # "POST_WITH_TEXT_ONLY",
    "POST_IN_HEBREW",
    "POST_IN_ENGLISH",
    "POST_IN_ARABIC",
    "POST_IN_OTHER",
    # "DAYS_FROM_ELECTION",
    # "DAYS_FROM_THREE_TEENAGER_KIDNAP",
    # "DAYS_FROM_PROTECTIVE_EDGE_OFFICIAL_START_DATE",
    # "DAYS_FROM_PROTECTIVE_EDGE_OFFICIAL_END_DATE",
    # "DAYS_FROM_DUMA_ARSON_ATTACK",
    # "DAYS_FROM_THIRD_INTIFADA_START_DATE",
    "DAYS_FROM_MK_BIRTHDAY",
    "POST_PUBLISHED_ON_SATURDAY",
    "COMMENT_PUBLISHED_ON_SATURDAY",
    # "NUM_OF_COMMENTS_BY_COMMENTATOR_ID_ON_GIVEN_MK_POSTS",
    "NUM_OF_LIKES_BY_COMMENTATOR_ID_ON_GIVEN_MK_POSTS",
    # "RATIO_OF_COMMENTS_BY_COMMENTATOR_ID_ON_GIVEN_MK_POSTS",
    "RATIO_OF_LIKES_BY_COMMENTATOR_ID_ON_GIVEN_MK_POSTS",
]

FEATURE_NAMES = TEXT_FEATURES + NUMERIC_FEATURES

y_vals = sorted(list(set(map(int, df[["ATTITUDE"]].get_values()))))
y_size = max(y_vals) + 1

LAYERS = 1
LAYER_DIM = 50
RNN_LAYERS = 1
RNN_LAYER_DIM = 50
WORD_EMBEDDING_DIM = 50
VOCAB_SIZE = 500
DROPOUT = .5
ITERATIONS = 100

m = dy.Model()
trainer = dy.AdamTrainer(m)

dims = [LAYER_DIM] * (LAYERS - 1)
in_dim = [len(TEXT_FEATURES) * RNN_LAYER_DIM + len(NUMERIC_FEATURES)] + dims
out_dim = dims + [y_size]
pW, pb = [list(map(m.add_parameters, d)) for d in (zip(out_dim, in_dim), out_dim)]
lstm = dy.LSTMBuilder(RNN_LAYERS, WORD_EMBEDDING_DIM, RNN_LAYER_DIM, m)
lookup = m.add_lookup_parameters((VOCAB_SIZE, WORD_EMBEDDING_DIM))

enum = {}


def encode(text):
    # TODO tokenize properly
    # TODO sort by frequency
    return [enum.setdefault(w, 1 + len(enum) if 1 + len(enum) < VOCAB_SIZE else 0) for w in text.split()]


def evaluate(features):
    dy.renew_cg()
    text_features, numeric_features = features[:len(TEXT_FEATURES)], features[len(TEXT_FEATURES):]
    for text in text_features:
        assert isinstance(text, str), "Not really text: %s" % text
    encoded = [encode(t) for t in text_features]
    state = lstm.initial_state()
    vectors = [[lookup[i] for i in f] for f in encoded]
    rnn_encoded = [state.transduce(v)[-1] if v else dy.inputVector(np.zeros(RNN_LAYER_DIM)) for v in vectors]
    x = dy.concatenate(rnn_encoded + [dy.inputVector(numeric_features)])
    for W, b in zip(*[map(dy.parameter, p) for p in (pW, pb)]):
        x = dy.dropout(x, DROPOUT)
        x = dy.tanh(W * x + b)
    return x


def predict(dataset):
    probs = []
    for sample in dataset[FEATURE_NAMES].get_values():
        out = evaluate(sample)
        y = dy.softmax(out).value()
        probs.append(y)
    y_pred = np.argmax(probs, axis=1)
    y = list(map(int, dataset[["ATTITUDE"]].get_values()))
    acc = np.mean(y_pred == y)
    print("accuracy = %.3f" % acc)


def learn():
    for iteration in range(ITERATIONS):
        print("iteration %d " % (iteration + 1))
        # np.random.shuffle(train)
        for i, sample in enumerate(train[FEATURE_NAMES + ["ATTITUDE"]].get_values()):
            out = evaluate(sample[:-1])
            loss = dy.pickneglogsoftmax(out, int(sample[-1]))
            loss.value()
            loss.backward()
            trainer.update()
        trainer.update_epoch()
        print("train ", end="")
        predict(train)
        print("test ", end="")
        predict(test)

learn()
