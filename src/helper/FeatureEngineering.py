from gensim.models.word2vec import Word2Vec
import numpy as np
import pickle
import os

os.chdir("../model")
gensim_model = Word2Vec.load(os.path.abspath(os.curdir)+"/word2vec.bin")

def get_vectors(text):
    all_vectors = []
    for token in text.split():
        vec = gensim_model[token]
        all_vectors.append(vec)

    vec_array = np.array(all_vectors)
    return vec_array.mean(axis=0)


def get_feature_data(data_frame):
    input = []
    target = []
    for row in data_frame.iterrows():
        question_vec = get_vectors(row[1]['question'])
        for idx, ans in enumerate(row[1]['sentences']):
            if len(ans) > 1:
                ans_vec = get_vectors(ans)
                input.append([question_vec, ans_vec])

                if idx in row[1]['candidates']:
                    target.append(1)
                else:
                    target.append(0)

    return input, target


def load_x_y(x_file, y_file):
    with open(x_file, "rb") as fp:
        X = pickle.load(fp)
        fp.close()

    with open(y_file, "rb") as fp:
        Y = pickle.load(fp)
        fp.close()

    return X, Y


def save_x_y(x, x_file, y, y_file):
    with open(x_file, "wb") as fp:
        pickle.dump(x, fp)
        fp.close()

    with open(y_file, "wb") as fp:
        pickle.dump(y, fp)
        fp.close()
