from nltk.corpus import wordnet
from itertools import product
import re


def check_if_dependency_exists(dep_tree, dependency):
    index = -1
    for idx, token in enumerate(dep_tree):
        if dependency in token:
            index = idx
    return index


def is_semantically_similar(word1, word2):
    syns1 = wordnet.synsets(word1)
    syns2 = wordnet.synsets(word2)
    all_scores = []
    for sense1, sense2 in product(syns1, syns2):
        all_scores.append(wordnet.wup_similarity(sense1, sense2))
    all_scores = list(filter(None, all_scores))
    if len(all_scores) > 0 and max(all_scores) >= 0.7:
        return True
    else:
        return False


def replace_preceded_compound(dep_tree, replace_with):
    pattern = u'(\w+)(\|)(compound)\s?(\w+)(\|)('+replace_with+')'
    search_result = re.search(pattern, dep_tree)
    if search_result is not None:
        return dep_tree.replace(search_result.group(), search_result.group(0).replace("compound", replace_with))
    else:
        return dep_tree


def replace_preceded_amod(dep_tree, replace_with):
    pattern = u'(\w+)(\|)(amod)\s?(\w+)(\|)('+replace_with+')'
    search_result = re.search(pattern, dep_tree)
    if search_result is not None:
        return dep_tree.replace(search_result.group(), search_result.group(0).replace("amod", replace_with))
    else:
        return dep_tree


def pre_processed_dependency(dep_tree):

    dep_tree = dep_tree.replace("||", " |")

    filtered_dep_tree = replace_preceded_compound(dep_tree, 'nsubj')
    filtered_dep_tree = replace_preceded_compound(filtered_dep_tree, 'pobj')
    filtered_dep_tree = replace_preceded_compound(filtered_dep_tree, 'dobj')

    filtered_dep_tree = replace_preceded_amod(filtered_dep_tree, 'nsubj')
    filtered_dep_tree = replace_preceded_amod(filtered_dep_tree, 'pobj')
    filtered_dep_tree = replace_preceded_amod(filtered_dep_tree, 'dobj')

    return filtered_dep_tree


if __name__ == '__main__':
    dep = "Harald|nsubjpass of|prep Norway|pobj and|cc Tostig|conj were|auxpass killed|ROOT ,|punct and|cc the|det Norwegians|nsubj suffered|conj such|amod great|amod losses|dobj that|dobj||xcomp only|advmod 24|nsubjpass of|prep the|det original|amod 300|nummod ships|pobj were|auxpass required|relcl to|aux carry|xcomp away|prt the|det survivors|dobj .|punct".replace("||", " |")
    print(dep)
    print(pre_processed_dependency(dep))

