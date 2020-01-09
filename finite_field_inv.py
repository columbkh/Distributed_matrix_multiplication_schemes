import numpy as np


FF_FIELD = 0
field = None


def getInvMatrix(rt, n, n_inv):
    x = [pow(rt, i, FF_FIELD) for i in range(n)]
    A_inv = [([1] + [x[(n - ((j * i) % n)) % n] for j in range(1, n)]) for i in range(n)]
    A = np.matrix(A_inv)
    A = A.getT()
    A_inv = np.asarray(A)
    Aff = matrix2ffmatrix(A_inv)
    n_invff = field(n_inv)
    return multMatrWithff(Aff, n_invff)


def getInvMatrix_with_x(x, n, n_inv):
    A_inv = [([1] + [x[(n - ((j * i) % n)) % n] for j in range(1, n)]) for i in range(n)]
    A = np.matrix(A_inv)
    A = A.getT()
    A_inv = np.asarray(A)
    Aff = matrix2ffmatrix(A_inv)
    n_invff = field(n_inv)
    return multMatrWithff(Aff, n_invff)


def multMatrWithff(matr, x):
    return [[col * x for col in row] for row in matr]


def matrix2ffmatrix(matr):
  #  print "matr", matr
   # for matr_str in matr:
   #     print "matr_str", matr_str
   #     for mat in matr_str:
    #        print "mat ", mat
    return [[field(mat) for mat in mat_str] for mat_str in matr]


def array2ffarray(arr):
    return [field(el) for el in arr]


def ff_power(f_element, n):
    ret = field(1)
    for i in range(n):
        ret = ret * f_element
    return ret


def transposeMatrix(m):
    return map(list, zip(*m))


def getMatrixMinor(m, i, j):
    return [row[:j] + row[j + 1:] for row in (m[:i] + m[i + 1:])]


def getMatrixDeternminant(m):
    if len(m) == 2:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]

    determinant = field(0)
    for c in range(len(m)):
        determinant += ff_power(field(-1), c) * m[0][c] * getMatrixDeternminant(getMatrixMinor(m, 0, c))
    return determinant


def getMatrixInverse(m):
    determinant = getMatrixDeternminant(m)
    if len(m) == 2:
        return [[m[1][1] / determinant, field.__neg__(m[0][1] / determinant)],
                [field.__neg__(m[1][0] / determinant), m[0][0] / determinant]]
    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = getMatrixMinor(m, r, c)
            cofactorRow.append(ff_power(field(-1), (r + c)) * getMatrixDeternminant(minor))
        cofactors.append(cofactorRow)
    cofactors = transposeMatrix(cofactors)
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c] / determinant
    return cofactors


def swap_rows(m, i, j):
    tmp = m[i]
    m[i] = m[j]
    m[j] = tmp
    return m


def add_row(m, i, j, elem):
    tmp = []
    for el in m[j]:
        tmp.append(elem * el)
    for count in range(len(m[i])):
        m[i][count] = m[i][count] + tmp[count]
    return m


def mult_coeff(m, i, coeff):
    for count in range(len(m[i])):
        m[i][count] = m[i][count] * coeff
    return m


def gaussian_elim(m):
    for i in range(len(m)):
        if m[i][i] == field(0):
            j = i + 1
            while m[i][j] == 0:
                j += 1
                if j >= len(m):
                    return -1
            m = swap_rows(m, i, j)
        m = mult_coeff(m, i, field.__invert__(m[i][i]))
        if i != 0:
            for count in range(i):
                m = add_row(m, count, i, field(-1) * m[count][i])
                if all(row == field(0) for row in m[count]):
                    return -1
        if i != len(m) - 1:
            for count in range(i + 1, len(m)):
                m = add_row(m, count, i, field(-1) * m[count][i])
                if all(row == field(0) for row in m[count]):
                    return -1
    return m


def inverse_matrix(m):
    ed = np.identity(len(m), dtype=int)
    ed_ff = matrix2ffmatrix(ed)
    app_m = [(m[count] + ed_ff[count]) for count in range(len(m))]
    gauss_m = gaussian_elim(app_m)
    if isinstance(gauss_m, int):
        if gauss_m == -1:
            return -1
    return [row[len(m):] for row in gauss_m]
