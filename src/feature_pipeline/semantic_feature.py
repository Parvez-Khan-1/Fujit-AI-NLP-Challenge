from src.helper import dependency_helper as dh
from collections import OrderedDict
import itertools
import spacy

nlp = spacy.load("en")


def get_dependency_tree(text):
    doc = nlp(text)
    info = []
    for token in doc:
        info.append(token.text+ "|"+token.dep_)
    return info


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
    comparision_matrix[0] = is_roots_similar(question_dep_dict, answer_dep_dict)
    comparision_matrix[1] = is_subj_similar(question_dep_dict, answer_dep_dict)
    comparision_matrix[2] = is_dobj_similar(question_dep_dict, answer_dep_dict)
    comparision_matrix[3] = is_pobj_similar(question_dep_dict, answer_dep_dict)

    return comparision_matrix


def is_semantically_similar(list1, list2):
    for (word1, word2) in itertools.product(list1, list2):
        if dh.is_semantically_similar(word1, word2):
            return True

    return False


def is_roots_similar(question_dep_dict, answer_dep_dict):
    if "ROOT" in question_dep_dict.keys() and "ROOT" in answer_dep_dict.keys():
        root1 = question_dep_dict.get("ROOT")
        root2 = answer_dep_dict.get("ROOT")

        if any(x in root1 for x in root2):
            return 1
        elif is_semantically_similar(root1, root2):
            return 1
        else:
            return 0

    else:
        return 0


def is_subj_similar(question_dep_dict, answer_dep_dict):
    if "nsubj" in question_dep_dict.keys() and "nsubj" in answer_dep_dict.keys():
        nsubj1 = question_dep_dict.get("nsubj")
        nsubj2 = answer_dep_dict.get("nsubj")
        if any(x in nsubj1 for x in nsubj2):
            return 1
        elif is_semantically_similar(nsubj1, nsubj2):
            return 1
        else:
            return 0

    else:
        return 0


def is_dobj_similar(question_dep_dict, answer_dep_dict):

    if "dobj" in question_dep_dict.keys() and "dobj" in answer_dep_dict.keys():
        dobj1 = question_dep_dict.get("dobj")
        dobj2 = answer_dep_dict.get("dobj")

        if any(x in dobj1 for x in dobj2):
            return 1
        elif is_semantically_similar(dobj1, dobj2):
            return 1
        else:
            return 0

    else:
        return 0


def is_pobj_similar(question_dep_dict, answer_dep_dict):

    if "pobj" in question_dep_dict.keys() and "pobj" in answer_dep_dict.keys():
        pobj1 = question_dep_dict.get("pobj")
        pobj2 = answer_dep_dict.get("pobj")
        if any(x in pobj1 for x in pobj2):
            return 1
        elif is_semantically_similar(pobj1, pobj2):
            return 1
        else:
            return 0
    else:
        return 0


def is_subj_obj_similar(dep_tree1, dep_tree2):
    index1 = dh.check_if_dependency_exists(dep_tree1, "pobj")
    index2 = dh.check_if_dependency_exists(dep_tree2, "pobj")
    if dh.check_if_dependency_exists(dep_tree1, "nsubj") and dh.check_if_dependency_exists(dep_tree2, "pobj"):
        if list(dep_tree1.keys())[list(dep_tree1.values()).index("nsubj")] == list(dep_tree2.keys())[list(dep_tree2.values()).index("pobj")]:
            return 1
        else:
            return 0
    else:
        return 0
