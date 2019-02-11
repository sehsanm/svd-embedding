import numpy as np
import os
import re
import src.evaluation

from scipy import sparse
from scipy.sparse.linalg import  svds
SPLIT_RE = re.compile('( |\\!|,|:|\\[|\\]|<|>|\\{|\\}|%|\\$|\\^|\\&|\\*|\\.|\\?|\\"|\\~|\\+|=،)+')


def load_corpus(corpus_file, index_dict, window_size=5):
    """This file will load the corpus file and creates the
    vocab index , as well as the  adjecency matrix of the corpus"""

    print('Loading the corpus')
    main_matrix = sparse.lil_matrix((len(index_dict), len(index_dict)))
    line_cnt = 0
    with open(corpus_file, 'r', encoding='UTF-8') as c_file:
        line = c_file.readline()
        while line:
            line = line.strip('\t\r\n ')
            line_cnt = line_cnt + 1
            if line_cnt % 10000 == 0:
                print('Lines processed: %s  : %s' , line_cnt,  line)
            if len(line) != 0:

                tokens = split_line(line)
                last_n = []
                for t in tokens:
                    if len(t) > 0 and t in index_dict:
                        for prev_token in last_n:
                            main_matrix[index_dict[t], index_dict[prev_token]] = main_matrix[index_dict[t], index_dict[
                                prev_token]] + 1

                        last_n.append(t)
                        if len(last_n) > window_size:
                            last_n.pop(0)
            line = c_file.readline()

    return main_matrix + main_matrix.T


def ppmi_inplace(matrix, k = 5 ):
    """Calculates the PPMI of the matrix and replace the matrix inplace"""

    print('Calculating PPMI')
    row_sum = matrix.sum(0)
    col_sum = matrix.sum(1)
    all_sum = matrix.sum()

    nnz_row, nnz_col = matrix.nonzero()

    for ind in range(len(nnz_col)):
        i = nnz_row[ind]
        j = nnz_col[ind]
        ppmi = max(0, np.log(matrix[i, j] * all_sum / (row_sum[0, i] * col_sum[j, 0])) - k)
        matrix[i, j] = ppmi

    return matrix


def calculate_svd(matrix, dim=300):
    print('Calculating SVDs')
    return svds(matrix, dim, return_singular_vectors='u')


def write_list_to_file(index, index_file):
    with open(index_file, 'w', encoding='UTF-8') as f:
        for l in index:
            f.write(l + '\n')


def load_vocab_index_from_file(index_input_file):
    """Loads the index file from previously built index file"""
    print("Loading the index file")
    index = 0
    ret = {}
    rev_index = []
    with open(index_input_file, 'r', encoding='UTF-8') as input:
        line = input.readline()
        while line:
            line = line.strip('\t\r\n ')
            ret[line] = index
            rev_index.append(line)
            index = index + 1
            line = input.readline()

    return ret, rev_index


def build_vocab_index_from_corpus(corpus_input_file, high_pass, low_pass):
    """This method builds the vocab index from corpus file"""

    print('Build Index from Corpus')
    vocab_freq = {}
    with open(corpus_input_file, 'r', encoding='UTF-8') as input:
        line = input.readline()
        while line:
            line = line.strip('\t\r\n ')
            if len(line) > 0:
                tokens = split_line(line)
                for t in tokens:
                    if len(t) > 0:
                        if t in vocab_freq:
                            vocab_freq[t] = vocab_freq[t] + 1
                        else:
                            vocab_freq[t] = 1
            line = input.readline()

    lst = list()
    for v, freq in vocab_freq.items():
        add = True
        if high_pass != None and freq > high_pass:
            add = False
        if low_pass != None and  freq < low_pass:
            add = False
        if add:
            lst.append(v)

    lst.sort()
    return lst


def split_line(line):
    """Splits the line in input. In future more complex tokenizers might be used"""
    tokens = SPLIT_RE.split(line)
    for t in tokens:
        t.strip('\\u200')
    return tokens


def build_model():
    if not os.path.isfile(index_file):
        index = build_vocab_index_from_corpus(corpus_file, None,  100)
        write_list_to_file(index, index_file)
    index_dict, rev_model = load_vocab_index_from_file(index_file)
    matrix = load_corpus(corpus_file, index_dict)
    ppmi_inplace(matrix)
    print('shape matrix is ')
    print(matrix.shape)
    u, s, vt = calculate_svd(matrix)
    np.savetxt(matrix_file, u * s)


def test_model():
    data_set = src.evaluation.load_analogy('../data/analogy.csv')
    index_dict, rev_index = load_vocab_index_from_file(index_file)
    print('Loading Matrix')
    embedding_matrix = np.loadtxt(matrix_file)

    #src.evaluation.run_analogy(embedding_matrix, index_dict, data_set, rev_index)
    test_lst = ['آمریکا' , 'ظریف' , 'خانه' ,'پول' , 'ایران' ]
    for t in test_lst:
        if t in index_dict:
            knn = src.evaluation.find_knn(embedding_matrix[index_dict[t]][:] , embedding_matrix)
            print(t)
            for x in range(20):
                print('\t' + rev_index[knn[x]])

if __name__ == "__main__":
    #base = 'sample'
    #base = 'bijan_khan_ut'
    base='wiki'
    index_file =  '../data/' + base + '-index.txt'
    matrix_file = '../data/' + base + '-vector.txt'
    corpus_file = '../data/' + base + '.txt'
    build_model()
    test_model()