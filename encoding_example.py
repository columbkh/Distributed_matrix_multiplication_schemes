import numpy as np
import tools
import time


def encode_An(lpart, i_plus, A, field, l, r, Zik):
    return [lpart[i] * ((A + sum([(pow(i_plus[i], k, field) * Zik[i][k - 1]) % field for k in range(1, l + 1)])) % field)
        for i in range(r)]


def encode_A(left_part, i_plus_an, A, field, N, l, r):
    Zik = [[np.matrix(np.random.random_integers(0, field-1, (A.shape[0], A.shape[1]))) for k in range(l)] for i in range(r)]
    return [encode_An(left_part[n], i_plus_an[n], A, field, l, r, Zik) for n in range(N)]


def encode_Bn(Bn, i_plus, field, l, r, Zik):
    return [(Bn[i] + sum([(pow(i_plus[i], k, field) * Zik[i][k - 1]) % field for k in range(1, l + 1)])) % field for i
            in range(r)]


def encode_B(Bn, i_plus_an, field, l, r, N):
    Zik = [[np.matrix(np.random.random_integers(0, field-1, (Bn[0].shape[0], Bn[0].shape[1]))) for k in range(l)] for i in
           range(r)]
    return [encode_Bn(Bn, i_plus_an[n], field, l, r, Zik) for n in range(N)]


print "Hello, world"
r = 14
N = 18
l = 2
field = 65537

A = np.matrix(np.random.random_integers(0, 255, (768, 768)))
B = np.matrix(np.random.random_integers(0, 255, (10752, 768)))

Bn = np.split(B, r)

d_cross, left_part, i_plus_an, an = tools.make_matrix_d_cross(N, field, r, l)
start = time.time()
Aenc = encode_A(left_part, i_plus_an, A, field, N, l, r)
pause = time.time()
Benc = encode_B(Bn, i_plus_an, field, l, r, N)
stop = time.time()
start_rand = time.time()
Zik = [[np.matrix(np.random.random_integers(0, field - 1, (A.shape[0], A.shape[1]))) for k in range(l)] for i in
       range(r)]
stop_rand = time.time()

start_pow = time.time()
for n in range(N):
    for i in range(r):
            sum([(pow(i_plus_an[n][i], k, field) * Zik[i][k - 1]) % field for k in range(1, l+1)]) % field
stop_pow = time.time()

print "A: ", pause - start
print "B: ", stop - pause
print "power: ", stop_pow - start_pow
print "random matrices: ", stop_rand - start_rand