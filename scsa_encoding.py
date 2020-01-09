from tools import *
import time
N = 18
field = 65537
l = 2
r = N - 2*l

def veryUnusualEncoding(A, m, n, i_plus_an, l, r, field, N):
    xs = []
    for alpha in i_plus_an:
        str = []
        for i in alpha:
            for k in range(l+1):
                str.append(pow(i, k, field))
        xs.append(str)
    xs_m = np.matrix(xs).transpose()
    A_encc = [[np.matrix(np.zeros((m, n), dtype=int)) for j in range(r)] for i in range(N)]
  #  return
    for zeile in range(m):
       for spalte in range(n):
           matr = []
           for str in range(r):
               matr.append(
                   [0]*str*(l+1) + [A[zeile, spalte]] +
                   [np.random.random_integers(0, field-1) for i in range(l)] +
                   [0]*((l+1)*(r - 1 - str))
               )
           res = (strassen_winograd(np.matrix(matr), xs_m) % field).tolist()
           for i in range(N):
               for j in range(r):
                   A_encc[i][j].itemset((zeile, spalte), res[j][i])
    return





if __name__ == "__main__":
    A = np.matrix(np.random.random_integers(0, field-1, (1000, 100)))
    d_cross, left_part, i_plus_an, an = make_matrix_d_cross(N, field, r, l)

    i_plus_an_trans = np.array(i_plus_an).transpose().tolist()

    start1 = time.time()
    Aenc = encode_A(left_part, i_plus_an, A, field, N, l, r)
    start2 = time.time()
    print "Time: ", start2 - start1
    start1 = time.time()
    veryUnusualEncoding(A, A.shape[0], A.shape[1], i_plus_an, l, r, field, N)
    start2 = time.time()
    print "Time: ", start2 - start1


