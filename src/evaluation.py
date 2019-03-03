import scipy
import numpy as np
from numpy import random

from scipy.spatial.distance import cosine

def load_analogy(file_name):
    """Loads the analogy dataset as a CSV file
    and returns a list of 5 element tuples (cat, word_1, word_2 , word_3, word_4)"""
    dataset = []
    with open(file_name, "r", encoding='UTF-8') as input_file:
        line = input_file.readline()
        while line:
            line = line.strip('\r\t\n ')
            parts = line.split(',')
            if len(parts) == 5:
                dataset.append((parts[0], parts[1], parts[2], parts[3], parts[4]))
            line = input_file.readline()
    return dataset


def sort_similarity(embedding_matrix, rev_index, word_1, word_2, word_3, answer_index):
    """Check the analogy """
    knn = find_knn(embedding_matrix[word_3][:] + embedding_matrix[word_2][:] - embedding_matrix[word_1][:]
                   , embedding_matrix , 'c')
    return np.where(knn == answer_index)[0][0]


def run_analogy(embedding_matrix, index_dictionary, analogy_dataset, rev_index , tol = 10):
    print('Running Analogy Test')
    correct_cat = {'total': 0}
    skipped_cat = {'total': 0}
    total_cat = {'total': 0}
    rand_ind = list(range(len(analogy_dataset)))
    random.shuffle(rand_ind)
    for r_ind in rand_ind:
        item = analogy_dataset[r_ind]
        if not item[0] in correct_cat:
            correct_cat[item[0]] = 0
        if not item[0] in skipped_cat:
            skipped_cat[item[0]] = 0
        if not item[0] in total_cat:
            total_cat[item[0]] = 0

        total_cat[item[0]] = total_cat[item[0]] + 1
        total_cat['total'] = total_cat['total'] + 1

        found = True
        if not item[1] in index_dictionary:
            found = False
            print(item[1], 'Not found')
        if not item[2] in index_dictionary:
            found = False
            print(item[2], 'Not found')
        if not item[3] in index_dictionary:
            found = False
            print(item[3], 'Not found')
        if not item[4] in index_dictionary:
            found = False
            print(item[4], 'Not found')
        if found:
            w_1_ind = index_dictionary[item[1]]
            w_2_ind = index_dictionary[item[2]]
            w_3_ind = index_dictionary[item[3]]
            w_4_ind = index_dictionary[item[4]]


            loc =  sort_similarity(embedding_matrix, rev_index, w_1_ind, w_2_ind, w_3_ind, w_4_ind)
            if loc < tol:
                print('Perfect ' + item[1] + ' to ' + item[2] + ' is like ' + item[3] + ' to ' + item[4])
                correct_cat[item[0]] = correct_cat[item[0]] + 1
                correct_cat['total'] = correct_cat['total'] + 1
            else:
                print('Failed for: ' +  item[1] + ' to ' + item[2] + ' is like ' + item[3] + ' to ' + item[4], ', AnswerIndex:' , loc)



        else:

            skipped_cat['total'] = skipped_cat['total'] + 1
            skipped_cat[item[0]] = skipped_cat[item[0]] + 1

        print('Accuracy so far:', correct_cat['total'] / total_cat['total'])

    print(total_cat)
    print(correct_cat)
    print(skipped_cat)

def cosine_dist(word_vec, embed_vec):
    return cosine(word_vec, embed_vec)


def euclid_dist(word_vec, embed_vec):
    return np.linalg.norm(word_vec - embed_vec)


def find_knn(word_vec, embedding_matrix, distance_method='c'):
    """Each row of the embedding matrix is the vector representation of words"""

    dists = []


    if distance_method == 'c':
        dists = 1 - np.dot(embedding_matrix , word_vec) / (np.linalg.norm(embedding_matrix , axis=1)  * np.linalg.norm(word_vec))

    return np.argsort(dists)

if __name__ == "__main__":
    pass


def print_closest(vec, embedding_matrix, rev_index, top_n):
    knn = find_knn(vec, embedding_matrix , 'e')
    for x in range(top_n):
        print('\t' + rev_index[knn[x]], '\t', cosine_dist(vec, embedding_matrix[knn[x]][:]))