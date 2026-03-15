from numba import jit
import numpy as np
from collections import Counter
from math import log
import argparse
import sys

def parseme():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train-data',
        help='A MSA in FASTA format to train a Hidden Markov Model',
        type=argparse.FileType('r'),
        required=True,
        metavar='FILE')
    parser.add_argument(
        '--test-data',
        help='Squence data in FASTA format with for which the \
        HHM Model has to calculate a score. If no data is provieded only the \
        emission_probabilities of the HMM are printed.',
        type=argparse.FileType('r'),
        metavar="FILE")
    parser.add_argument(
        '--out',
        help='Path to output File. Only raw data is printed (Default: stdout).',
        type=argparse.FileType('w'),
        default=sys.stdout,
        metavar='FILE')

    args = parser.parse_args()
    return args.train_data, args.test_data, args.out

class HMM:
    def __init__(self, MSA):
        self.MSA = MSA
        self.MSAbool = self.boolify(MSA)
        self.MSAchar = [char for char in np.unique(MSA) if char != '-']
        self.alignment_number, self.alignment_length = MSA.shape
        self.match_states = self.calc_match_states()
        self.n = Counter(self.match_states)[True]

        self.transmissions = self.calc_transmissions()
        self.emissions_from_M, self.emissions_from_I = self.calc_emissons()

    def viterbi(self, testdata):
        char_to_int = {c: i for i, c in enumerate(self.MSAchar)}
        testdata = np.array([char_to_int[c] for c in testdata.strip()])

        return self._viterbi(
            testdata,
            self.emissions_from_M,
            self.emissions_from_I,
            self.transmissions,
            self.n,
        )

    @staticmethod
    @jit(nopython=True)
    def _viterbi(x, e_M, e_I, a, L):
        N = x.size
        V_M = np.zeros((N, L+1))
        V_I = np.zeros((N, L+1))
        V_D = np.zeros((N, L+1))

        for i in range(N):
            for j in range(1, L+1):
                V_M[i, j] = log(e_M[x[i]][j-1]) + \
                    max(V_M[i-1][j-1] + log(a[0][j-1]), # M->M
                        V_I[i-1][j-1] + log(a[1][j-1]), # M->D
                        V_D[i-1][j-1] + log(a[2][j-1])) # M->I

                V_I[i, j] = log(e_I[x[i]][j-1]) + \
                    max(V_M[i-1][j] + log(a[3][j]), # I->M
                        V_I[i-1][j] + log(a[4][j]), # I->I
                        V_D[i-1][j] + log(a[5][j])) # I->D

                V_D[i, j] = max(V_M[i][j-1] + log(a[6][j-1]), # D->M
                                V_I[i][j-1] + log(a[7][j-1]), # D->D
                                V_D[i][j-1] + log(a[8][j-1])) # D->I

        return max(V_M[N-1, L], V_I[N-1, L], V_D[N-1, L])


    def calc_match_states(self):
        return [Counter(column)['-'] < self.alignment_number//2 for column in self.MSA.T]

    def equal_parts(self, l, n):
        for i in range(0, len(l), n):
            yield l[i:i+n]

    @staticmethod
    @jit
    def boolify(array2d):
        boolarray = np.zeros(array2d.shape, dtype=np.bool_)
        for i in range(array2d.shape[0]):
            for j in range(array2d.shape[1]):
                boolarray[i, j] = array2d[i, j] != '-'
        return boolarray

    def calc_emissons(self):
        emi_M = {char: np.zeros(self.n) for char in self.MSAchar}
        emi_I = {char: np.zeros(self.n+1) for char in self.MSAchar}

        match_index = 0
        for alignment, match in zip(self.MSA.T, self.match_states):
            char_count = Counter(alignment)
            if match:
                for char in self.MSAchar:
                    emi_M[char][match_index] = char_count[char]
                match_index += 1
            if not match:
                for char in self.MSAchar:
                    emi_I[char][match_index] += char_count[char]

        for c in self.MSAchar:
            emi_M[c] += np.ones(self.n)
            emi_I[c] += np.ones(self.n+1)

        M_sum = np.sum(list(emi_M.values()), axis=0)
        I_sum = np.sum(list(emi_I.values()), axis=0)

        for c in self.MSAchar:
            emi_M[c] /= M_sum
            emi_I[c] /= I_sum

        return \
            np.vstack([emi_M[c] for c in self.MSAchar]), \
            np.vstack([emi_I[c] for c in self.MSAchar])

    def calc_transmissions(self):

        trans_list = [
            'M->M', 'M->D', 'M->I', 'I->M', 'I->I', 'I->D', 'D->M', 'D->D', 'D->I']
        trans = {t: np.zeros(self.n+1) for t in trans_list}

        first_row_char_count = Counter(self.MSAbool.T[0])
        if self.match_states[0]:
            trans['M->M'][0] = first_row_char_count[True]
            trans['M->D'][0] = first_row_char_count[False]
            trans['M->I'][0] = 0
        else:
            trans['I->M'][0] = first_row_char_count[True]
            trans['I->I'][0] = first_row_char_count[False]

        m_idel = 1
        for i, (alignment, alignment_next, match, match_next) in \
                enumerate(
                zip(self.MSAbool.T[:-1],    self.MSAbool.T[1:],
                    self.match_states[:-1], self.match_states[1:])):
            trans_cnt = Counter(zip(alignment, alignment_next))

            if match and match_next:
                for booltrans, transm in zip(
                    [(True, True), (True, False), (False, True), (False, False)],
                    ['M->M',        'M->D',        'D->M',        'D->D']):
                    trans[transm][m_idel] = trans_cnt[booltrans]
                m_idel += 1

            if match and not match_next:
                next_match = self.match_states.index(True, i)
                empty_til_next = [not any(rest)for rest in self.MSAbool[:, i+2:next_match]]
                trans_cnt = Counter(zip(alignment_next, empty_til_next))
                for booltrans, transm in zip(
                    [(True, True), (True, False), (False, True), (False, False)],
                    ['M->M',        'M->I',        'D->M',        'D->I']):
                    trans[transm][m_idel] += trans_cnt[booltrans]

            if not match and not match_next:
                for booltrans, transm in zip(
                    [(True, True), (True, False)],
                    ['I->I',        'I->M']):
                    trans[transm][m_idel] += trans_cnt[booltrans]

            if not match and match_next:
                for booltrans, transm in zip(
                    [(True, True), (True, False)],
                    ['I->M',        'I->D']):
                    trans[transm][m_idel] += trans_cnt[booltrans]
                m_idel += 1

            if m_idel == self.n:
                if i + 2 == self.alignment_length:
                    trans['M->M'][m_idel] = self.alignment_number
                else:
                    empty_end = [not any(rest) for rest in self.MSAbool[:, i+2:]]
                    trans_cnt = Counter(zip(alignment_next, empty_end))
                    for booltrans, transm in zip(
                        [(True, True), (True, False), (False, True), (False, False)],
                        ['M->M',        'M->I',        'D->M',        'D->I']):
                        trans[transm][
                            m_idel] = trans_cnt[booltrans]
                    trans['I->M'][m_idel] = self.alignment_number - \
                        trans['M->M'][m_idel]
                break

        for t in trans_list:
            trans[t] += np.ones(self.n+1)

        for t1, t2, t3 in self.equal_parts(trans_list, 3):
            abs_occur = trans[t1] + \
                trans[t2] + trans[t3]
            trans[t1] /= abs_occur
            trans[t2] /= abs_occur
            trans[t3] /= abs_occur
            
        return np.vstack([trans[t] for t in trans_list]) 