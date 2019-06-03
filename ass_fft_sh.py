import argparse
from tools import *
import sys
import numpy as np
import time
from mpi4py import MPI

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--Field', type=int, help='Finite Field')
parser.add_argument('--N', type=int, help='number of workers')
parser.add_argument('--l', type=int, help='l')
parser.add_argument('--barrier', help='Enable barrier', action="store_true")
parser.add_argument('--verific', help='Enable Verification', action="store_true")
parser.add_argument('--all_together', help='Compute all together', action="store_true")

args = parser.parse_args()


def ass_fft_m():


if find_fft_field(args.Field) == False:
    print "Wrong field"
    sys.exit(100)
else:
    F = args.Field
    possbs = get_all_possabilities(args.N)
    if not possbs:
        print "No possabilities"
        sys.exit(100)

    else:
        #		possb = np.random.random_integers(0, len(possbs)-1)
        tmp = 0
        possb = possbs[0]
        for pos in possbs:
            if pos.l > tmp:
                tmp = pos.l
            possb = pos
        # print possb

        r_a = possb.r_a
        r_b = possb.r_b
        N = possb.N
        l = possb.l
        k = possb.k
        # print r_a, r_b, N, l, k

        rts = find_rt(F, k)
        rt = rts[np.random.random_integers(0, len(rts) - 1)]

if args.barrier:
    barrier = True
else:
    barrier = False

if args.verific:
    verific = True
else:
    verific = False

if args.all_together:
    together = True
else:
    together = False

m = 1000
n = 1000
p = 1000

if m % r_a != 0:
    m = (m // r_a) * r_a
if p % r_b != 0:
    p = (p // r_b) * r_b

comm = MPI.COMM_WORLD

if comm.rank == 0:  # Master
    print "Running with %d processes:" % comm.Get_size()

    print "m: ", m
    print "p: ", p
    print "n: ", n
print "r_a: ", r_a
print "r_b: ", r_b
print "N: ", N
print "l: ", l
print "k: ", k
print "rt: ", rt

A = np.matrix(np.random.random_integers(0, 255, (m, n)))
B = np.matrix(np.random.random_integers(0, 255, (p, n)))

Ap = np.split(A, r_a)
Bp = np.split(B, r_b)

Ka = [np.matrix(np.random.random_integers(0, 255, (m / r_a, n))) for i in range(l)]
Kb = [np.matrix(np.random.random_integers(0, 255, (p / r_b, n))) for i in range(l)]

dec_start = time.time()

x = [pow(rt, i, F) for i in range(k)]
t = 3
for i in range(N - k):
    while is_power2(t) == True:
        t += 1
x.append(t)

Aenc = getAenc(Ap, Ka, N, F, l, r_a, x)
Benc = getBenc(Bp, Kb, N, F, l, r_a, r_b, x)

dec_pause = time.time()
dec_firstpart = dec_pause - dec_start

Rdict = []
for i in range(N):
    Rdict.append(np.zeros((m / r_a, p / r_b), dtype=np.int_))
Crtn = []
serv_comp = [None] * N
for i in range(N):
    Crtn.append(np.zeros((m / r_a, p / r_b), dtype=np.int_))

reqA = [None] * N
reqB = [None] * N
reqC = [None] * N
reqS = [None] * N

if together:
    ul_start = time.time()
    for i in range(N):
        reqA[i] = comm.Isend([Aenc[i], MPI.INT], dest=i + 1, tag=15)
        reqB[i] = comm.Isend([Benc[i], MPI.INT], dest=i + 1, tag=29)
        reqC[i] = comm.Irecv([Rdict[i], MPI.INT], source=i + 1, tag=42)

    MPI.Request.Waitall(reqA)
MPI.Request.Waitall(reqB)

ul_stop = time.time()
ul = ul_stop - ul_start

if barrier:
    comm.Barrier()

dl_start = time.time()

lst = []
for i in range(k):
    j = MPI.Request.Waitany(reqC)
    lst.append(j)
    Crtn[j] = Rdict[j]
dl_stop = time.time()

missing = set(range(k)) - set(lst)

dl = dl_stop - dl_start

else:
# lalalala
ul_start = [None] * N
ul_stop = [None] * N
ul = [None] * N
dl = [None] * N
reqAB = [None] * 2 * N
for i in range(N):
    ul_start[i] = time.time()
    reqAB[i] = comm.Isend([Aenc[i], MPI.INT], dest=i + 1, tag=15)
    reqAB[i + N] = comm.Isend([Benc[i], MPI.INT], dest=i + 1, tag=29)
    reqC[i] = comm.Irecv([Rdict[i], MPI.INT], source=i + 1, tag=42)

lsst = []
lst = []

for i in range(N * 2):
    j = MPI.Request.Waitany(reqAB)
    lsst.append(j)
    if j < N:
        if j + N in lsst:
            ul_stop[j] = time.time()
    else:
        if j - N in lsst:
            ul_stop[j - N] = time.time()

for i in range(N):
    ul[i] = ul_stop[i] - ul_start[i]

if barrier:
    comm.Barrier()

dl_start = time.time()

for i in range(k):
    j = MPI.Request.Waitany(reqC)
    tmp = time.time()
    dl[j] = tmp - dl_start
    lst.append(j)
    Crtn[j] = Rdict[j]

while None in dl:
    dl.remove(None)

missing = set(range(k)) - set(lst)

dec_pause = time.time()

interpol(missing, Crtn, F, k, lst, x)
res = inverse_fft(k, Crtn[:k], rt, F)

final_res = []

for k_tmp in range(r_b):
    final_res += res[k_tmp * (r_a + l): (k_tmp + 1) * (r_a + l) - l]

dec_done = time.time()
dec_secondpart = dec_done - dec_pause
dec = dec_firstpart + dec_secondpart

#    print "Time spent calculating the result is: %f" % (bp_result - bp_received)

if barrier:
    comm.Barrier()

for i in range(N):
    serv_comp[i] = comm.recv(source=i + 1, tag=64)

print_times(dec, dl, ul, serv_comp)

if verific:
    Cver = []
    for bb in Bp:
        Cver += [(aa * bb.getT()) % F for aa in Ap]

print ([np.array_equal(final_res[i], Cver[i]) for i in range(len(Cver))])

else:  # Worker
Ai = np.empty_like(np.matrix([[0] * n for i in range(m / r_a)]))
Bi = np.empty_like(np.matrix([[0] * n for i in range(p / r_b)]))
rA = comm.Irecv(Ai, source=0, tag=15)
rB = comm.Irecv(Bi, source=0, tag=29)

rA.wait()
rB.wait()

print("Worker " + str(comm.rank) + " received the message from master")

servcomp_start = time.time()

Ci = (Ai * (Bi.getT())) % F

servcomp_done = time.time()

servcomp = servcomp_done - servcomp_start

if barrier:
    comm.Barrier()

sC = comm.Isend(Ci, dest=0, tag=42)
sC.Wait()
print("Worker " + str(comm.rank) + " sent the result to master")

if barrier:
    comm.Barrier()

sSrv = comm.send(servcomp, dest=0, tag=64)
