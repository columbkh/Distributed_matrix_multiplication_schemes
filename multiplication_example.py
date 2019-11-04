import numpy as np
import tools
import time

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



print "Hello, world"
r = 14
N = 18
l = 2
field = 65537

k = 4

LEAF_SIZE = 50
size = LEAF_SIZE * 2**k

A = [np.matrix(np.random.random_integers(0, 255, (size, size))) for i in range(r)]
B = [np.matrix(np.random.random_integers(0, 255, (size, size))) for i in range(r)]

atop, adown = np.split(A[0], 2)
a11, a12 = np.split(atop, 2, axis=1)
a21, a22 = np.split(adown, 2, axis=1)

Ci = np.matrix([[0] * size for i in range(size)])
Ci_str = np.matrix([[0] * size for i in range(size)])

start_str = time.time()
for j in range(r):
    Ci_str += strassen(A[j], B[j])
    Ci_str = Ci_str % field
    print "iteration ", j
stop_str = time.time()

start = time.time()
for j in range(r):
    Ci += A[j] * B[j]
    Ci = Ci % field
    print "iteration ", j
stop = time.time()


print "Multiplication: ", stop - start
print "Strassen Multiplication: ", stop_str - start_str
print "Relation: ", (stop - start) / (stop_str - start_str)

print ([np.array_equal(Ci[i], Ci_str[i]) for i in range(r)])
