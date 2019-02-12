import scipy
import numpy as np

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


def check_analogy(embedding_matrix, rev_index , word_1, word_2, word_3, answer_index, tol=10):
    """Check the analogy """
    knn = find_knn(embedding_matrix[word_3][:] + embedding_matrix[word_2][:] - embedding_matrix[word_1][:]
                   , embedding_matrix)
    if answer_index in knn[0:tol]:
        return True
    else:
        # print(rev_index[word_1] + ':' + rev_index[word_2] + ':' + rev_index[word_3])
        # for ind in knn[0:tol]:
        #     print('\t' + rev_index[ind])
        return False


def run_analogy(embedding_matrix, index_dictionary, analogy_dataset, rev_index):
    print('Running Analogy Test')
    correct_cat = {'total': 0}
    skipped_cat = {'total': 0}
    total_cat = {'total': 0}
    for item in analogy_dataset:
        if not item[0] in correct_cat:
            correct_cat[item[0]] = 0
        if not item[0] in skipped_cat:
            skipped_cat[item[0]] = 0
        if not item[0] in total_cat:
            total_cat[item[0]] = 0

        total_cat[item[0]] = total_cat[item[0]] + 1
        total_cat['total'] = total_cat['total'] + 1

        if item[1] in index_dictionary and \
                item[2] in index_dictionary and \
                item[3] in index_dictionary and \
                item[4] in index_dictionary:
            w_1_ind = index_dictionary[item[1]]
            w_2_ind = index_dictionary[item[2]]
            w_3_ind = index_dictionary[item[3]]
            w_4_ind = index_dictionary[item[4]]

            if check_analogy(embedding_matrix, rev_index,  w_1_ind, w_2_ind, w_3_ind, w_4_ind):
                print('Perfect ' + item[1] + ' to ' + item[2] + ' is like ' + item[3] + ' to ' + item[4])
                correct_cat[item[0]] = correct_cat[item[0]] + 1
                correct_cat['total'] = correct_cat['total'] + 1



        else:
            skipped_cat['total'] = skipped_cat['total'] + 1
            skipped_cat[item[0]] = skipped_cat[item[0]] + 1

    print(total_cat)
    print(correct_cat)
    print(skipped_cat)

def cosine_dist(word_vec, embed_vec):
    return cosine(word_vec, embed_vec)


def find_knn(word_vec, embedding_matrix, distance_method='c'):
    """Each row of the embedding matrix is the vector representation of words"""

    dists = []

    for row in range(len(embedding_matrix)):
        if distance_method == 'c':
            dists.append(cosine_dist(word_vec, embedding_matrix[row][:]))
    return np.argsort(dists)

if __name__ == "__main__":
    pass