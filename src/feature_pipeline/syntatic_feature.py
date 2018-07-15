from py_stringmatching import OverlapCoefficient, TfIdf, Jaccard
from sklearn.metrics.pairwise import cosine_similarity



oc = OverlapCoefficient()
jac = Jaccard()


def overlap_coefficient(text1, text2):
    return oc.get_sim_score(text1.split(), text2.split())
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

def initialize_tf_idf(row):
    question = row[1]['question']
    answers = list(filter(None, row[1]['sentences']))
    documents =  [question.split()]
    for answer in answers:
        documents.append(answer.split())
    tf_idf = TfIdf(documents)

    return tf_idf


def get_tf_idf(tf_idf, text1, text2):
    return tf_idf.get_sim_score(text1.split(), text2.split())


def jaccard_coefficient(text1, text2):
    return jac.get_sim_score(text1.split(), text2.split())


def get_cosine_similarity(question_matrix, answer_matrix):
    return cosine_similarity(answer_matrix, question_matrix)[0][0]

