import pandas as pd
from sklearn.cross_validation import train_test_split
import pickle


def read_json(JSON_PATH):
    try:
        return pd.read_json(JSON_PATH, lines=True)
    except FileNotFoundError as e:
        print(e)
        return None


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



def split_data_set_to_train_test_using_scikit(data_frame):
    train, test = train_test_split(data_frame, test_size=0.3)
    return train, test
