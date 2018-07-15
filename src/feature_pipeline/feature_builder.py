import nltk
from collections import OrderedDict
from src.feature_pipeline import syntatic_feature
from src.feature_pipeline import semantic_feature
from src.helper import pre_processing
from src.helper import dependency_helper as dh


def create_qa_vocab(row):
    question = row[1]['question']
    answers = row[1]['sentences']
    vocab = set(question.split())
    for answer in answers:
        for word in answer.split():
            vocab.add(word)
    return vocab


def create_qa_vocab_without_stopwords(row):
    question = pre_processing.remove_custom_stop_words(row[1]['question'])
    answers = row[1]['sentences']
    vocab = set(question.split())
    for answer in answers:
        for word in pre_processing.remove_custom_stop_words(answer).split():
            vocab.add(word)
    return vocab


def get_pos_tags(text):
    pos_tag_list = nltk.pos_tag(nltk.word_tokenize(text))
    pos_tokens = []
    for token in pos_tag_list:
        pos_tokens.append(token[0] + "|" + token[1])
    return pos_tokens


def create_pos_qa_vocab(row):
    question = row[1]['question']
    vocab = set(get_pos_tags(question))
    answers = row[1]['sentences']
    for answer in answers:
        pos_tag_list = get_pos_tags(answer)
        vocab.update(set(pos_tag_list))
    return vocab


def extract_feature(row):
    overlap_coefficients = []
    tf_idf_scores = []
    tf_idf = syntatic_feature.initialize_tf_idf(row)
    for answer in row[1]['sentences']:
        overlap_coefficients.append(syntatic_feature.overlap_coefficient(row[1]['question'], answer))
        tf_idf_scores.append(syntatic_feature.get_tf_idf(tf_idf, row[1]['question'], answer))
    return overlap_coefficients, tf_idf_scores


def get_occurence_matrix(vocab, answers):
    occurrence_matrix = []
    for token in vocab:
        if token in answers:
            occurrence_matrix.append(1)
        else:
            occurrence_matrix.append(0)
    return occurrence_matrix


def get_max_scores(scores_list):
    m = max(scores_list)
    return [i for i, j in enumerate(scores_list) if j == m]


def get_template(row):
    result_dict = OrderedDict({"section": row[1]['section'], "question": row[1]['question'],
                               "candidates": row[1]['candidates'], "sentences": row[1]['sentences'],
                               "article": row[1]['article'], "type": row[1]['type'],
                               "is_paraphrase": row[1]['is_paraphrase']})
    return result_dict


def get_syntactic_features(tf_idf, question, answer, composite_vocab, composite_pos_vocab, question_matrix, question_pos_matrix, question_matrix_without_stop_words, composite_vocab_without_stop_words):

    occurrence_matrix = get_occurence_matrix(composite_vocab, answer.split())

    occurrence_pos_matrix = get_occurence_matrix(composite_pos_vocab, answer.split())

    occurrence_matrix_without_stopwords = get_occurence_matrix(composite_vocab_without_stop_words, pre_processing.remove_custom_stop_words(answer).split())

    cosine_similarity = syntatic_feature.get_cosine_similarity([question_matrix], [occurrence_matrix])

    cosine_pos_similarity = syntatic_feature.get_cosine_similarity([question_pos_matrix], [occurrence_pos_matrix])

    cosine_similarity_without_stopwords = syntatic_feature.get_cosine_similarity([question_matrix_without_stop_words], [occurrence_matrix_without_stopwords])

    overlap_coefficient = syntatic_feature.overlap_coefficient(question, answer)

    jaccard_coefficient = syntatic_feature.jaccard_coefficient(question, answer)

    tf_idf_score = syntatic_feature.get_tf_idf(tf_idf, question, answer)

    return [cosine_similarity, cosine_pos_similarity, overlap_coefficient, jaccard_coefficient, tf_idf_score, cosine_similarity_without_stopwords]


def get_syntactic_features_(tf_idf, question, answer, composite_vocab, composite_pos_vocab, question_matrix, question_pos_matrix):

    occurrence_matrix = get_occurence_matrix(composite_vocab, answer.split())

    occurrence_pos_matrix = get_occurence_matrix(composite_pos_vocab, answer.split())

    cosine_similarity = syntatic_feature.get_cosine_similarity([question_matrix], [occurrence_matrix])

    cosine_pos_similarity = syntatic_feature.get_cosine_similarity([question_pos_matrix], [occurrence_pos_matrix])

    overlap_coefficient = syntatic_feature.overlap_coefficient(question, answer)

    jaccard_coefficient = syntatic_feature.jaccard_coefficient(question, answer)

    tf_idf_score = syntatic_feature.get_tf_idf(tf_idf, question, answer)

    return [cosine_similarity, cosine_pos_similarity, overlap_coefficient, jaccard_coefficient, tf_idf_score]


def get_dependency_dict(text):
    dep_tree = semantic_feature.get_dependency_tree(text)
    dep_tree = dh.pre_processed_dependency(' '.join(dep_tree))
    dep_dict = semantic_feature.create_dict_from_dep_tree(dep_tree)
    return dep_dict


def get_semantic_features(question, answer):
    question_dep_dict = get_dependency_dict(question)
    ans_dep_dict = get_dependency_dict(answer)
    dep_comp_matrix = semantic_feature.get_comparision_matrix(question_dep_dict, ans_dep_dict)
    return dep_comp_matrix




