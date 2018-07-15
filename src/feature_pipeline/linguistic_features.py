import spacy
from src.feature_pipeline import semantic_feature
from collections import OrderedDict


nlp = spacy.load("en")

target_dependencies = ['ROOT', 'pobj', 'nsubj', 'amod', 'compound', 'dobj']


def get_linguistic_info(text):
    doc = nlp(text)
    parse_tree = OrderedDict({"NA":"NA"})
    for token in doc:
        if token.dep_ in parse_tree.keys():
            parse_tree = parse_tree.get(token.dep_).append(token)
        else:
            parse_tree[token.dep_] = [token.dep]
    return parse_tree


def get_linguistic_info_(text):
    doc = nlp(text)
    info = []
    for token in doc:
        info.append(token.text+ "|"+token.dep_)
    return info


def get_correct_ans_list(row):
    candidates = row[1]['candidates']
    answers = row[1]['sentences']
    correct_ans = []
    for each_candidate in candidates:
        correct_ans.append(answers[each_candidate])
    return correct_ans


def get_all_ans_list(row):
    answers = row[1]['sentences']
    correct_ans = []
    for idx, answer in enumerate(answers):
        correct_ans.append(answers[idx])
    return correct_ans


def get_token_type(token):
    return token.split('|')[1]


def filter_dependency_tree_old(deep_tree):
    filtered_dict = {}
    for key, value in deep_tree.items():
        if value in target_dependencies:
            filtered_dict[key] = value
    return filtered_dict


def create_dict_from_dep_tree(dep_tree):
    dep_tree_dict = OrderedDict()
    for idx, token in enumerate(dep_tree.split()):
        word, dep = token.split('|')
        if idx == 0:
            dep_tree_dict[dep] = [word]
        elif dep in dep_tree_dict:
            dep_tree_dict.get(dep).append(word)
        else:
            dep_tree_dict[dep] = [word]

    return dep_tree_dict


def get_comparision_matrix(question_dep_dict, answer_dep_dict):
    comparision_matrix = [0, 0, 0, 0]
    comparision_matrix[0] = semantic_feature.is_roots_similar(question_dep_dict, answer_dep_dict)
    comparision_matrix[1] = semantic_feature.is_subj_similar(question_dep_dict, answer_dep_dict)
    comparision_matrix[2] = semantic_feature.is_dobj_similar(question_dep_dict, answer_dep_dict)
    comparision_matrix[3] = semantic_feature.is_pobj_similar(question_dep_dict, answer_dep_dict)

    return comparision_matrix

