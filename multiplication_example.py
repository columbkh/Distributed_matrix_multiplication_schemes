import numpy as np
import tools
import time

def cutoff_criterium(m, k, n):
    return 1.0 <= 200.0 * (4.0/n + 4.0/m + 7.0/k)

def strassen(A, B):
    n = len(A)

    if n <= LEAF_SIZE:
        return A * B
    top, down = np.split(A, 2)
    a11, a12 = np.split(top, 2, axis=1)
    a21, a22 = np.split(down, 2, axis=1)

    top, down = np.split(B, 2)
    b11, b12 = np.split(top, 2, axis=1)
    b21, b22 = np.split(down, 2, axis=1)

    # Calculating p1 to p7:
    p1 = strassen(a11+a22, b11+b22) # p1 = (a11+a22) * (b11+b22)

    p2 = strassen(a21+a22, b11)  # p2 = (a21+a22) * (b11)

    p3 = strassen(a11, b12 - b22)  # p3 = (a11) * (b12 - b22)

    p4 = strassen(a22, b21 - b11)  # p4 = (a22) * (b21 - b11)

    p5 = strassen(a11+a12, b22)  # p5 = (a11+a12) * (b22)

    p6 = strassen(a21-a11, b11+b12)  # p6 = (a21-a11) * (b11+b12)

    p7 = strassen(a12-a22, b21+b22)  # p7 = (a12-a22) * (b21+b22)

    # calculating c21, c21, c11 e c22:
    c12 = p3 + p5  # c12 = p3 + p5
    c21 = p2 + p4  # c21 = p2 + p4
    c11 = p1 + p4 - p5 + p7  # c11 = p1 + p4 - p5 + p7
    c22 = p1 + p3 - p2 + p6  # c22 = p1 + p3 - p2 + p6

    top = np.concatenate((c11, c12), axis=1)
    down = np.concatenate((c21, c22), axis=1)
    C = np.concatenate((top, down))
    return C

def strassen_winograd(A, B):
    n, k = A.shape
    k, m = B.shape


    if cutoff_criterium(m, k, n):
        return A * B

  #  if n <= LEAF_SIZE_N or k <= LEAF_SIZE_K or m<= LEAF_SIZE_M:
  #      return A * B
    if (m % 2 != 0) and (n % 2 != 0) and (k % 2 != 0):
        Atop, Adown = np.split(A, [n-1])
        A11, A12 = np.split(Atop, [k-1], axis=1)
        A21, A22 = np.split(Adown, [k-1], axis=1)

        Btop, Bdown = np.split(B, [k-1])
        B11, B12 = np.split(Btop, [m-1], axis=1)
        B21, B22 = np.split(Bdown, [m-1], axis=1)

        C11 = strassen_winograd(A11, B11)
        a12_b21 = np.dot(A12, B21)
        C11 += a12_b21
        C12 = np.dot(A11, B12) + np.dot(A12, B22)
        C21 = np.dot(A21, B11) + np.dot(A22, B21)
        C22 = np.dot(A21, B12) + np.dot(A22, B22)

        top_c = np.concatenate((C11, C12), axis=1)
        down_c = np.concatenate((C21, C22), axis=1)
        C = np.concatenate((top_c, down_c))
        return C

    if (n % 2 != 0) and (k % 2 != 0) and (m % 2 == 0):
        Atop, Adown = np.split(A, [n - 1])
        A11, A12 = np.split(Atop, [k - 1], axis=1)
        A21, A22 = np.split(Adown, [k - 1], axis=1)

        B11, B21 = np.split(B, [k - 1])

        C11 = strassen_winograd(A11, B11)
        a12_b21 = np.dot(A12, B21)
        C11 += a12_b21
        C21 = np.dot(A21, B11) + np.dot(A22, B21)

        C = np.concatenate((C11, C21))
        return C

    if (m % 2 != 0) and (k % 2 != 0) and (n % 2 == 0):
        A11, A12 = np.split(A, [k - 1], axis=1)
        Btop, Bdown = np.split(B, [k - 1])
        B11, B12 = np.split(Btop, [m - 1], axis=1)
        B21, B22 = np.split(Bdown, [m - 1], axis=1)

        C11 = strassen_winograd(A11, B11)
        a12_b21 = np.dot(A12, B21)
        C11 += a12_b21
        C12 = np.dot(A11, B12) + np.dot(A12, B22)

        C = np.concatenate((C11, C12), axis=1)
        return C

    if (m % 2 != 0) and (n % 2 != 0) and (k % 2 == 0):
        A11, A21 = np.split(A, [n-1])
        B11, B12 = np.split(B, [m - 1], axis=1)

        C11 = strassen_winograd(A11, B11)
        C12 = np.dot(A11, B12)
        C21 = np.dot(A21, B11)
        C22 = np.dot(A21, B12)

        top_c = np.concatenate((C11, C12), axis=1)
        down_c = np.concatenate((C21, C22), axis=1)
        C = np.concatenate((top_c, down_c))
        return C

    if (n % 2 != 0) and (m % 2 == 0) and (k % 2 == 0):
        A11, A21 = np.split(A, [n-1])

        C11 = strassen_winograd(A11, B)
        C21 = np.dot(A21, B)

        C = np.concatenate((C11, C21))
        return C

    if (m % 2 != 0) and (n % 2 == 0) and (k % 2 == 0):
        B11, B12 = np.split(B, [m - 1], axis=1)

        C11 = strassen_winograd(A, B11)
        C12 = np.dot(A, B12)

        C = np.concatenate((C11, C12), axis=1)
        return C

    if (k % 2 != 0) and (m % 2 == 0) and (n % 2 == 0):
        A11, A12 = np.split(A, [k - 1], axis=1)
        B11, B21 = np.split(B, [k - 1])

        C = strassen_winograd(A11, B11)
        a12_b21 = np.dot(A12, B21)
        C += a12_b21

        return C


    top, down = np.split(A, 2)
    a11, a12 = np.split(top, 2, axis=1)
    a21, a22 = np.split(down, 2, axis=1)

    top, down = np.split(B, 2)
    b11, b12 = np.split(top, 2, axis=1)
    b21, b22 = np.split(down, 2, axis=1)

    # Calculating p1 to p7:
    s1 = a21 + a22

    s2 = s1 - a11

    s3 = a11 - a21

    s4 = a12 - s2

    t1 = b12 - b11

    t2 = b22 - t1

    t3 = b22 - b12

    t4 = b21 - t2

    p1 = strassen_winograd(a11, b11) # p1 = (a11+a22) * (b11+b22)

    p2 = strassen_winograd(a12, b21)  # p2 = (a21+a22) * (b11)

    p3 = strassen_winograd(s1, t1)  # p3 = (a11) * (b12 - b22)

    p4 = strassen_winograd(s2, t2)  # p4 = (a22) * (b21 - b11)

    p5 = strassen_winograd(s3, t3)  # p5 = (a11+a12) * (b22)

    p6 = strassen_winograd(s4, b22)  # p6 = (a21-a11) * (b11+b12)

    p7 = strassen_winograd(a22, t4)  # p7 = (a12-a22) * (b21+b22)

    u1 = p1 + p2

    u2 = p1 + p4

    u3 = u2 + p5

    u4 = u3 + p7

    u5 = u3 + p3

    u6 = u2 + p3

    u7 = u6 + p6


    top = np.concatenate((u1, u7), axis=1)
    down = np.concatenate((u4, u5), axis=1)
    C = np.concatenate((top, down))
    return C


print "Hello, world"
r = 14
N = 18
l = 2
field = 65537

kk = 4

LEAF_SIZE_N = 64
LEAF_SIZE_K = 32
LEAF_SIZE_M = 32

size_n = LEAF_SIZE_N * 2**kk + 5
size_k = LEAF_SIZE_K * 2**kk + 3
size_m = LEAF_SIZE_M * 2**kk + 10

print "anf k", size_k
print "anf n", size_n
print "anf m", size_m


print "size_k", size_k
print "size_n", size_n
print "size_m", size_m

A = [np.matrix(np.random.random_integers(0, 255, (size_n, size_k))) for i in range(r)]
B = [np.matrix(np.random.random_integers(0, 255, (size_k, size_m))) for i in range(r)]

#print A

Ci = np.matrix([[0] * size_m for i in range(size_n)])
#Ci_str = np.matrix([[0] * size for i in range(size)])
Ci_win = np.matrix([[0] * size_m for i in range(size_n)])
#Ci_add = np.matrix([[0] * size for i in range(size)])


start_win = time.time()
for j in range(r):
    print "allle"
    Ci_win += strassen_winograd(A[j], B[j])
    Ci_win = Ci_win % field
    print "iteration ", j
stop_win = time.time()


start = time.time()
for j in range(r):
    Ci += A[j] * B[j]
    Ci = Ci % field
    print "iteration ", j
stop = time.time()

print "Standard Multiplication: ", stop - start
print "Strassen-Winograd Multiplication: ", stop_win - start_win
#print "Relation: ", (stop - start) / (stop_str - start_str)

#print ([np.array_equal(Ci[i], Ci_str[i]) for i in range(r)])
print ([np.array_equal(Ci[i], Ci_win[i]) for i in range(r)])
