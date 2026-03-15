import math
import numpy as np
from pip._vendor.distlib.compat import raw_input
import time

def create_topology(M):
    np.set_printoptions(suppress=True)
    N = M + M + 1 + M + 2
    states = []
    states.append("S")
    if M > 0:
        states.append("I0")
    for i in range(M):
        states.append("M%s" % (i + 1))
        states.append("I%s" % (i + 1))
        states.append("D%s" % (i + 1))
    states.append("E")
    return states

def initialize_parameters(M,X,T,msa_sequences,acceptable_columns,binary_matrix):
    np.set_printoptions(suppress=True)
    N = M + M + 1 + M + 2
    b_matrix = np.zeros((N, T))
    a_matrix = np.zeros((N, N))
    pi_vector = np.zeros((N))

    profile_dic = {}
    pseudocount = 0.04
    for i in range(M):
        gap_row = 0
        z=0
        for x in X:
            profile_dic[x] = 0
        for seq in msa_sequences:
            if seq[acceptable_columns[i]] != '-':
                profile_dic[seq[acceptable_columns[i]]] = profile_dic[seq[acceptable_columns[i]]] + 1
            else:
                gap_row = gap_row + 1

        for t in range(T-1):
            b_matrix[((i+1)*3)-1, t] = (profile_dic[X[t]] + pseudocount) / (((T-1) * pseudocount) + (len(msa_sequences) - gap_row))

    for i in range(M):
        for t in range(T-1):
            b_matrix[1, t] = 1 / (T - 1)
            b_matrix[(i+1)*3, t] = 1 / (T-1)
        pi_vector[i] = 0

    for i in range(M):
        for t in range(T):
            if X[t] == '-':
                b_matrix[((i+1)*3)+1, t] = 1

    for t in range(T):
        if X[t] == '-':
            b_matrix[0, t] = 1
            b_matrix[N-1, t] = 1

    pi_vector[0] = 1

    a_matrix[0, 1] = 1/3
    a_matrix[0, 2] = 1/3
    a_matrix[0, 4] = 1/3
    # match states
    for i in range(1,M):
        a_matrix[(i*3)-1,((i+1)*3)-1] = 0.95
        a_matrix[(i*3)-1,i*3] = 0.25
        a_matrix[(i*3)-1,((i+1)*3)+1] = 0.25
    a_matrix[(M * 3) - 1, M*3] = 0.95
    a_matrix[(M * 3) - 1, N-1] = 0.5

    # insertion states
    a_matrix[1,1] = 0.77
    a_matrix[1,2] = 0.23
    for i in range(1,M):
        a_matrix[i*3,i*3] = 0.77
        a_matrix[i*3,((i+1)*3)-1] = 0.23
    a_matrix[M*3, M*3] = 0.77
    a_matrix[M*3, N-1] = 0.23

    # deletion states
    for i in range(1,M):
        a_matrix[(i*3)+1,((i+1)*3)+1] = 0.67
        a_matrix[(i*3)+1,((i+1)*3)-1] = 0.33
    a_matrix[(M * 3) + 1, N-1] = 1

    return a_matrix,b_matrix,pi_vector


def forward_algorithm(M,X,a_matrix,pi_vector,b_matrix,seq):
    np.set_printoptions(suppress=True)
    T = len(seq)
    N = M + M + 1 + M + 2
    alpha = np.zeros((N,T))

    for i in range(N):
        alpha[i,0] = pi_vector[i] * b_matrix[i,X.index(seq[0])]
    #     induction
    for t in range(1,T):
        for j in range(N):

            s = 0
            for i in range(N):
                s = s + alpha[i,t-1] * a_matrix[i,j]

            alpha[j, t] = s * b_matrix[j,X.index(seq[t])]

    return alpha

def backward_algorithm (M, X, a_matrix, b_matrix, seq):
    np.set_printoptions(suppress=True)
    N = M + M + 1 + M + 2
    T = len(seq)
    beta = np.zeros((N,T))

    for i in range(N):
        beta[i,T-1] = 1

    for t in reversed(range(T-1)):
        for i in range(N):
            for j in range(N):
                beta[i,t] = beta[i,t] + (a_matrix[i,j] * b_matrix[j,X.index(seq[t+1])] * beta[j,t+1])
    return beta

# learning problem
def gamma_S (M,X,a_matrix,pi_vector,b_matrix,alpha,beta,seq):
    np.set_printoptions(suppress=True)
    N = M + M + 1 + M + 2
    T = len(seq)

    gama = np.zeros((N,T))
    S = []
    for t in range(T):
        S.append(np.zeros((N,N)))

    for t in range(T):
        for i in range(N):
            gama[i,t] = alpha[i,t] * beta[i,t]

    sum_last_column = 0
    for k in range(N):
        sum_last_column = sum_last_column + alpha[k,T-1]

    for t in range(T-1):
        for i in range(N):
            for j in range(N):
                S[t][i, j] = alpha[i, t] * a_matrix[i, j] * b_matrix[j, X.index(seq[t + 1])] * beta[j, t + 1]
                if sum_last_column != 0:
                    S[t][i, j] = S[t][i, j]/sum_last_column

    return gama,S


def learning_algorithm(gamma_list,S_list,X,M , a_matrix,pi_vector, b_matrix,msa_sequences):
    np.set_printoptions(suppress=True)
    N = M + M + 1 + M + 2

    for i in range(N):
        for j in range(N):
            y = 0
            for n in range(len(msa_sequences)):
                z = 0
                L = len(msa_sequences[n])
                S = S_list[n]
                for t in range(L - 1):
                    z = S[t][i, j] + z
                y = y + z

            f = 0
            for n in range(len(msa_sequences)):
                z = 0
                L = len(msa_sequences[n])
                gamma = gamma_list[n]
                for t in range(L - 1):
                    z = z + gamma[i, t]
                f = f + z

            if f != 0:
                a_matrix[i, j] = y / f

    for j in range(N):
        for t in range(len(X)):
            if b_matrix[j, t] != 0:
                y = 0
                for n in range(len(msa_sequences)):
                    L = len(msa_sequences[n])
                    gamma = gamma_list[n]
                    z = 0
                    for o in range(L):
                        if seq[o] == X[t]:
                            z = gamma[j, o] + z

                    y = y + z

                f = 0
                for n in range(len(msa_sequences)):
                    L = len(msa_sequences[n])
                    gamma = gamma_list[n]
                    z = 0
                    for o in range(L):
                        z = gamma[j, o] + z
                    f = f + z

                if f != 0:
                    b_matrix[j, t] = y / f

    return a_matrix,b_matrix

def update_matrices(mat, b_matrix,T,M,X):
    np.set_printoptions(suppress=True)
    N = M + M + 1 + M + 2
    for i in range(M):
        for t in range(T - 1):
            b_matrix[1, t] = 1 / (T - 1)
            b_matrix[(i + 1) * 3, t] = 1 / (T - 1)
        pv[i] = 0

    for i in range(M):
        for t in range(T):
            if X[t] == '-':
                b_matrix[((i + 1) * 3) + 1, t] = 1

    profile_dic = {}
    pcnt = 0.4

    for i in range(M):
        s = 0
        for t in range(T - 1):
            if b_matrix[((i + 1) * 3) - 1, t] == 0:
                b_matrix[((i + 1) * 3) - 1, t] = pcnt
            s = b_matrix[((i + 1) * 3) - 1, t] + s
        for t in range(T - 1):
            b_matrix[((i + 1) * 3) - 1, t] = b_matrix[((i + 1) * 3) - 1, t] / s

    ps = 0.004
    s = 0
    if mat[0, 1] == 0:
        mat[0,1] = ps
    s = s + mat[0,1]
    if mat[0, 2] == 0:
        mat[0,2] = ps
    s = s + mat[0,2]
    if mat[0, 4] == 0:
        mat[0,4] = ps
    s = s + mat[0,4]
    mat[0, 1] = mat[0, 1]/s
    mat[0, 2] = mat[0, 2]/s
    mat[0, 4] = mat[0, 4]/s


    # match states
    for i in range(1, M):
        if max(mat[(i * 3) - 1, ((i + 1) * 3) - 1], mat[(i * 3) - 1, i * 3], mat[(i * 3) - 1, ((i + 1) * 3) + 1])!= mat[(i * 3) - 1, ((i + 1) * 3) - 1]:

            mat[(i * 3) - 1, ((i + 1) * 3) - 1] = 0.95
            mat[(i * 3) - 1, i * 3] = 0.25
            mat[(i * 3) - 1, ((i + 1) * 3) + 1] = 0.25

        s = 0
        if mat[(i * 3) - 1, ((i + 1) * 3) - 1] == 0:
            mat[(i * 3) - 1, ((i + 1) * 3) - 1] = ps
        s = s + mat[(i * 3) - 1, ((i + 1) * 3) - 1]
        if mat[(i * 3) - 1, i * 3] == 0:
            mat[(i * 3) - 1, i * 3] = ps
        s = s + mat[(i * 3) - 1, i * 3]
        if mat[(i * 3) - 1, ((i + 1) * 3) + 1] == 0:
            mat[(i * 3) - 1, ((i + 1) * 3) + 1] = ps
        s = s + mat[(i * 3) - 1, ((i + 1) * 3) + 1]
        mat[(i * 3) - 1, ((i + 1) * 3) - 1] = mat[(i * 3) - 1, ((i + 1) * 3) - 1] / s
        mat[(i * 3) - 1, i * 3] = mat[(i * 3) - 1, i * 3] / s
        mat[(i * 3) - 1, ((i + 1) * 3) + 1] = mat[(i * 3) - 1, ((i + 1) * 3) + 1] / s


    if max(mat[(M * 3) - 1, M * 3],mat[(M * 3) - 1, N - 1]) != mat[(M * 3) - 1, M * 3]:
        mat[(M * 3) - 1, M * 3] = 0.95
        mat[(M * 3) - 1, N - 1] = 0.5


    s = 0
    if mat[(M * 3) - 1, M * 3] == 0:
        mat[(M * 3) - 1, M * 3] = ps
    s = s + mat[(M * 3) - 1, M * 3]
    if mat[(M * 3) - 1, N - 1] == 0:
        mat[(M * 3) - 1, N - 1] = ps
    s = s + mat[(M * 3) - 1, N - 1]
    mat[(M * 3) - 1, M * 3] = mat[(M * 3) - 1, M * 3] / s
    mat[(M * 3) - 1, N - 1] = mat[(M * 3) - 1, N - 1] / s

    # insertion states
    if max(mat[1,1],mat[1,2]) != mat[1, 1]:
        mat[1, 1] = 0.77
        mat[1, 2] = 0.23

    s = 0
    if mat[1, 1] == 0:
        mat[1, 1] = ps
    s = s + mat[1, 1]
    if mat[1, 2] == 0:
        mat[1, 2] = ps
    s = s + mat[1, 2]
    mat[1, 1] = mat[1, 1] / s
    mat[1, 2] = mat[1, 2] / s


    for i in range(1, M):
        if max(mat[i * 3, i * 3] ,mat[i * 3, ((i + 1) * 3) - 1]) != mat[i * 3, i * 3]:
            mat[i * 3, i * 3] = 0.77
            mat[i * 3, ((i + 1) * 3) - 1] = 0.23

        s = 0
        if mat[i * 3, i * 3] == 0:
            mat[i * 3, i * 3] = ps
        s = s + mat[i * 3, i * 3]
        if mat[i * 3, ((i + 1) * 3) - 1] == 0:
            mat[i * 3, ((i + 1) * 3) - 1] = ps
        s = s + mat[i * 3, ((i + 1) * 3) - 1]
        mat[i * 3, i * 3] = mat[i * 3, i * 3] / s
        mat[i * 3, ((i + 1) * 3) - 1] = mat[i * 3, ((i + 1) * 3) - 1] / s

    if max(mat[M * 3, M * 3],mat[M * 3, N - 1]) != mat[M * 3, M * 3]:
        mat[M * 3, M * 3] = 0.77
        mat[M * 3, N - 1] = 0.23

    s = 0
    if mat[M * 3, M * 3] == 0:
        mat[M * 3, M * 3] = ps
    s = s + mat[M * 3, M * 3]
    if mat[M * 3, N - 1] == 0:
        mat[M * 3, N - 1] = ps
    s = s + mat[M * 3, N - 1]
    mat[M * 3, M * 3] = mat[M * 3, M * 3] / s
    mat[M * 3, N - 1] = mat[M * 3, N - 1] / s

    # deletion states
    for i in range(1, M):

        if max(mat[(i * 3) + 1, ((i + 1) * 3) + 1],mat[(i * 3) + 1, ((i + 1) * 3) - 1]) != mat[(i * 3) + 1, ((i + 1) * 3) + 1]:
            mat[(i * 3) + 1, ((i + 1) * 3) + 1] = 0.67
            mat[(i * 3) + 1, ((i + 1) * 3) - 1] = 0.33

        s = 0
        if mat[(i * 3) + 1, ((i + 1) * 3) + 1] == 0:
            mat[(i * 3) + 1, ((i + 1) * 3) + 1] = ps
        s = s + mat[(i * 3) + 1, ((i + 1) * 3) + 1]
        if mat[(i * 3) + 1, ((i + 1) * 3) - 1] == 0:
            mat[(i * 3) + 1, ((i + 1) * 3) - 1] = ps
        s = s + mat[(i * 3) + 1, ((i + 1) * 3) - 1]
        mat[(i * 3) + 1, ((i + 1) * 3) + 1] = mat[(i * 3) + 1, ((i + 1) * 3) + 1] / s
        mat[(i * 3) + 1, ((i + 1) * 3) - 1] = mat[(i * 3) + 1, ((i + 1) * 3) - 1] / s


    mat[(M * 3) + 1, N - 1] = 1

    return mat,b_matrix

def viterbi_algorithm(M,X,a_matrix,pi_vector,b_matrix,seq,aligned_seq):
    np.set_printoptions(suppress=True)
    N = M + M + 1 + M + 2
    T = len(seq)

    path_list = []
    max_path_list = []
    path_list_item = []
    next_char = 0
    delta = pi_vector[0] * b_matrix[0, X.index(seq[0])]
    data_tuple = ('0', seq[next_char], '0', delta)
    path_list.append(data_tuple)
    max_path_list.append(data_tuple)
    path_list_item.append(data_tuple)
    subs_list = []
    while True:
        item = path_list_item[0]
        char_list = []
        t_list = []
        index_list = []
        delta_list = []
        summ = 0
        i = int(item[0])
        for j in range(N - 1):
            delta = 0
            if a_matrix[i, j] != 0:
                if ((j - 1) % 3 == 0 or j == N - 1) and j != 1:
                    t = int(item[2])
                    character = item[1] + '-'
                else:
                    t = int(item[2]) + 1
                    character = item[1] + seq[t]

                max_list = []
                for e in path_list:
                    if int(e[2]) == t - 1:
                        max_list.append(e[3] * a_matrix[int(e[0]), j])
                if len(max_list) != 0:
                    delta = max(max_list) * b_matrix[j, X.index(character[-1])]
                if delta == 0:
                    delta = 0.04

                index_list.append(j)
                char_list.append(character)
                t_list.append(t)
                delta_list.append(delta)

        if len(subs_list) != 0:
            for s in subs_list:
                summ = s[3] + summ
        for delta in delta_list:
            summ = delta + summ

        if len(subs_list) != 0:
            list_sub=[]
            for s in subs_list:
                delta = s[3] / summ
                data_tuple = (s[0], s[1], s[2], s[3] / summ)
                # subs_list.remove(s)
                list_sub.append(data_tuple)

            subs_list.clear()
            for s in list_sub:
                subs_list.append(s)

        for f in range(len(delta_list)):
            if summ != 0:
                data_tuple = (index_list[f], char_list[f], t_list[f], delta_list[f]/summ)
            subs_list.append(data_tuple)

        path_list_item.remove(item)
        t = int(item[2])+1
        for s in subs_list:
            if s[2] < t and s[1].count("-") < 3 and s[3] != 0:
                    path_list_item.append(s)
                    subs_list.remove(s)
        if len(path_list_item) == 0:
            max_index = []
            for s in subs_list:
                max_index.append(s[3])
            max_delta = max(max_index)
            index = max_index.index(max_delta)
            if subs_list[index][1].count("-") >=4:
                subs_list.remove(subs_list[index])
                max_index = []
                for s in subs_list:
                    max_index.append(s[3])
                max_delta = max(max_index)
                index = max_index.index(max_delta)

            max_path_list.append(subs_list[index])
            path_list_item.append(subs_list[index])
            for s in subs_list:
                path_list.append(s)
            subs_list.clear()

        if len(max_path_list[-1][1]) - max_path_list[-1][1].count("-") == len(seq)-2:
            return max_path_list[-1][1][1:]
            break


start_time = time.time()
np.set_printoptions(suppress=True)
dir_address = " "
dir_address = raw_input()
if dir_address == " ":
    dir_address = raw_input("Enter dir of test cases: ")

msa_sequences = []
f = open(dir_address+"/in/input.txt", "r")
n, m = f.readline().split()
for msa in range(int(n)):
    msa_sequences.append("-"+f.readline()[0:-2]+"-")

test_seq = "-"+f.readline()[:-1]+"-"

T = 0
X = []
seq_type = ''
characters = []
for seq in msa_sequences:
    for i in seq:
        if i not in characters:
            characters.append(i)

T = 5
X = ['A', 'T', 'G', 'C', '-']
acceptable_columns = []

for i in range(len(msa_sequences[0])):
    gap_number = 0
    for j in msa_sequences:
        if j[i] == '-':
            gap_number = gap_number + 1

    if gap_number < int(m):
        acceptable_columns.append(i)

M = len(acceptable_columns)
N = M + M + 1 + M + 2
b_matrix = np.zeros((N, T))
a_matrix = np.zeros((N, N))
pv = np.zeros((N))
binary_matrix = np.zeros((N, N))
states = []
states = create_topology(M)
a_matrix, b_matrix, pv = initialize_parameters(M, X, T, msa_sequences, acceptable_columns, binary_matrix)

for i in range(10):
    alpha_list = []
    beta_list = []
    gamma_list = []
    S_list = []
    C_list = []
    C = np.zeros((T))
    S = []
    for s in range(len(msa_sequences)):
        alpha = np.zeros((N, T))
        beta = np.zeros((N, T))
        gamma = np.zeros((N, T))

        alpha = forward_algorithm(M, X, a_matrix, pv, b_matrix, msa_sequences[s])
        alpha_list.append(alpha)

        beta = backward_algorithm(M, X, a_matrix, b_matrix, msa_sequences[s])
        beta_list.append(beta)

        gamma, S = gamma_S(M, X, a_matrix, pv, b_matrix, alpha, beta, msa_sequences[s])
        gamma_list.append(gamma)
        S_list.append(S)

    a_matrix, b_matrix = learning_algorithm(gamma_list, S_list, X, M, a_matrix, pv, b_matrix, msa_sequences)
    a_matrix, b_matrix = update_matrices(a_matrix, b_matrix, T, M, X)
    pv[0] = 1

q_correct = " "
q = " "
q = viterbi_algorithm(M, X, a_matrix, pv, b_matrix, test_seq, msa_sequences[0])
f_output = open(dir_address+"/out/output.txt", "r")
print(test_seq[1:-1]+" "+q_correct+" "+q)
end_time = time.time()
print(f"\n⏱️ 程序运行时间：{end_time - start_time:.2f} 秒")