import os
import tflearn
import json
import tqdm
import numpy as np
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_1d
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical
from src.helper import data_helper
from src.feature_pipeline import feature_builder
from src.feature_pipeline import syntatic_feature
from src.helper import Evaluation
from src.helper import pre_processing
from src.helper import FeatureEngineering


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train_network(x, y, x_dev, y_dev):
    network = input_data(shape=[None, 9, 1])

    network = conv_1d(network, 32, 3, activation='relu')

    network = conv_1d(network, 64, 3, activation='relu')

    network = conv_1d(network, 64, 3, activation='relu')

    # network = fully_connected(network, 512, activation='relu')

    # network = dropout(network, 0.5)

    network = fully_connected(network, 2, activation='sigmoid')

    network = regression(network, optimizer='momentum', learning_rate=0.001)

    model = tflearn.DNN(network, tensorboard_verbose=0)

    model.fit(x, y, n_epoch=5, validation_set=(x_dev, y_dev), show_metric=True)

    return model


def make_predictions(test_data, model):
    actual = []
    predicted = []
    f = open('SelQA-ass-result.example.json', 'w')
    for row in test_data.iterrows():
        result_dict = feature_builder.get_template(row)
        composite_vocab = feature_builder.create_qa_vocab(row)
        composite_pos_vocab = feature_builder.create_pos_qa_vocab(row)

        question = pre_processing.do_pre_processing(row[1]['question'])
        question_matrix = feature_builder.get_occurence_matrix(composite_vocab, question.split())
        question_pos_matrix = feature_builder.get_occurence_matrix(composite_pos_vocab, question.split())

        tf_idf = syntatic_feature.initialize_tf_idf(row)
        predictions = []

        for idx, answer in enumerate(row[1]['sentences']):
            if len(answer) > 1:
                answer = pre_processing.do_pre_processing(answer)
                dep_comp_matrix = feature_builder.get_semantic_features(question, answer)
                syn_features = feature_builder.get_syntactic_features_(tf_idf, question, answer, composite_vocab, composite_pos_vocab, question_matrix, question_pos_matrix)
                X = syn_features + dep_comp_matrix
                X = np.reshape(X, [-1, 9, 1])
                prediction = model.predict(X)
                predictions.append(float(prediction[0][1]))
            else:
                predictions.append(0.0)

        predicted.append(feature_builder.get_max_scores(predictions))
        result_dict["results"] = predictions
        f.write(json.dumps(result_dict))
        f.write('\n')
        actual.append(row[1]['candidates'])
        print("Predicted Result :", feature_builder.get_max_scores(predictions))
        print("Actual Answer :", row[1]['candidates'])
        print("-"*40)

    print("Result file have been stored at data/SelQA-ass-result.example.json")
    return actual, predicted


def create_x_y(data_frame):
    X = []
    Y = []
    for row in tqdm.tqdm(data_frame.iterrows()):
        composite_vocab = feature_builder.create_qa_vocab(row)
        composite_pos_vocab = feature_builder.create_pos_qa_vocab(row)

        question = pre_processing.do_pre_processing(row[1]['question'])
        question_matrix = feature_builder.get_occurence_matrix(composite_vocab, (question).split())
        question_pos_matrix = feature_builder.get_occurence_matrix(composite_pos_vocab, (question).split())

        tf_idf = syntatic_feature.initialize_tf_idf(row)
        for idx, answer in enumerate(row[1]['sentences']):
            if len(answer) > 1:
                answer = pre_processing.do_pre_processing(answer)
                dep_comp_matrix = feature_builder.get_semantic_features(question, answer)
                syn_features = feature_builder.get_syntactic_features_(tf_idf, question, answer, composite_vocab, composite_pos_vocab, question_matrix, question_pos_matrix)
                X.append(syn_features + dep_comp_matrix)
                if idx in row[1]['candidates']:
                    Y.append(1)
                else:
                    Y.append(0)
    return X, Y


def reshape_x_y(x, y):
    x = np.reshape(x, [-1, 9, 1])
    y = to_categorical(y, nb_classes=2)
    y = np.reshape(y, (-1, 2))
    return x, y


if __name__ == '__main__':
    os.chdir("../data")
    train_data = data_helper.read_json(os.path.abspath(os.curdir) + '/SelQA-ass-train.json')
    test_data = data_helper.read_json(os.path.abspath(os.curdir) + '/SelQA-ass-test.json')
    dev_data = data_helper.read_json(os.path.abspath(os.curdir) + '/SelQA-ass-dev.json')

    # X, Y = create_x_y(dev_data)
    # FeatureEngineering.save_x_y(X, os.path.abspath(os.curdir)+"/x_dev_new.txt", Y, os.path.abspath(os.curdir)+"/y_dev_new.txt")

    X, Y = FeatureEngineering.load_x_y(os.path.abspath(os.curdir)+"/x_train.txt", os.path.abspath(os.curdir)+"/y_train.txt")
    X_dev, Y_dev = FeatureEngineering.load_x_y(os.path.abspath(os.curdir)+"/x_dev.txt", os.path.abspath(os.curdir)+"/y_dev.txt")

    X, Y = reshape_x_y(X, Y)
    X_dev, Y_dev = reshape_x_y(X_dev, Y_dev)

    print(X.shape)
    print(Y.shape)

    model = train_network(X, Y, X_dev, Y_dev)
    actual, predicted = make_predictions(test_data, model)

    os.chdir("../model")
    model.save(os.path.abspath(os.curdir)+"/tf_cnn.tfl")

