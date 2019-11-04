from tools import *
from mpi4py import *
import time
import sys
import communicators

d_cross = None
left_part = None
i_plus_an = None
an = None
Zik_a = None
Zik_b = None
summe_a = []
summe_b = []


def schema1(A, B, l, r, N, left_part, i_plus_an, field):
    Bn = np.split(B, r)

    start = time.time()
    Aenc = [[left_part[n][i] * ((A + summe_a[n][i]) % field) for i in range(r)] for n in range(N)]
    Benc = [[(Bn[i] + summe_b[n][i]) for i in range(r)] for n in range(N)]

#    Aenc, time_a = test_encode_A(left_part, i_plus_an, A, field, N, l, r)
 #   Benc, time_b = test_encode_B(Bn, i_plus_an, field, l, r, N)



    stop = time.time()

    random = 0.0

    return [A], Bn, Aenc, Benc, stop - start, random

def schema2(A, B, l, r, N, left_part, i_plus_an, field):
    An = np.split(A, r)

    start = time.time()

    Aenc = reverse_encode_A(An, i_plus_an, field, l, r, N)
    Benc = reverse_encode_B(left_part, i_plus_an, B, field, N, l, r)

    stop = time.time()
    return An, [B], Aenc, Benc, stop - start

def scs_m(N, l, r, field, barrier, verific, together, A, B, m, p, flazhok, i):
    if communicators.prev_comm.rank == 0:  # Master

        if N > 19:
            print "Too many instances"
            sys.exit(100)

        global Zik_a, Zik_b, d_cross, left_part, i_plus_an, an, summe_a, summe_b
        tools_for_enc_start = time.time()

        if i == 0:
            print "lol"
            d_cross, left_part, i_plus_an, an = make_matrix_d_cross(N, field, r, l)
            Zik_a = [[np.matrix(np.random.random_integers(0, field - 1, (A.shape[0], A.shape[1]))) for k in range(l)] for
                   i in range(r)]
            Zik_b = [[np.matrix(np.random.random_integers(0, field - 1, (B.shape[0] / r, B.shape[1]))) for k in range(l)] for
                   i in range(r)]

            summe_a = []
            summe_b = []

            for n in range(N):
                summe_n = []
                summe_n_b = []
                for i in range(r):
                    summe_n.append(sum([(pow(i_plus_an[n][i], k, field) * Zik_a[i][k - 1]) % field for k in range(1, l + 1)]) % field)
                    summe_n_b.append(sum([(pow(i_plus_an[n][i], k, field) * Zik_b[i][k - 1]) % field for k in range(1, l + 1)]) % field)
                summe_a.append(summe_n)
                summe_b.append(summe_n_b)

        tools_for_enc_stop = time.time()
        tools_for_enc_time = tools_for_enc_stop - tools_for_enc_start
        random = None

        if flazhok:
            if r == 1:
                An, Bn, Aenc, Benc, enc_time = schema1(A, B, l, r, N, left_part, i_plus_an, field)
            else:
                if m <= p:
                    An, Bn, Aenc, Benc, enc_time, random = schema1(A, B, l, r, N, left_part, i_plus_an, field)
                else:
                    An, Bn, Aenc, Benc, enc_time = schema2(A, B, l, r, N, left_part, i_plus_an, field)
        else:
            if r == 1:
                An, Bn, Aenc, Benc, enc_time = schema2(A, B, l, r, N, left_part, i_plus_an, field)
            else:
                if m <= p:
                    An, Bn, Aenc, Benc, enc_time = schema2(A, B, l, r, N, left_part, i_plus_an, field)
                else:
                    An, Bn, Aenc, Benc, enc_time = schema1(A, B, l, r, N, left_part, i_plus_an, field)


        Crtn = []
        serv_comp = [None] * N

        for i in range(N):
            if flazhok:
                if r == 1:
                    Crtn.append(np.zeros((m, p / r), dtype=np.int_))
                else:
                    if m <= p:
                        Crtn.append(np.zeros((m, p / r), dtype=np.int_))
                    else:
                        Crtn.append(np.zeros((m / r, p), dtype=np.int_))
            else:
                if r == 1:
                    Crtn.append(np.zeros((m / r, p), dtype=np.int_))
                else:
                    if m <= p:
                        Crtn.append(np.zeros((m / r, p), dtype=np.int_))
                    else:
                        Crtn.append(np.zeros((m, p / r), dtype=np.int_))

        req_a = [None] * N * r
        req_b = [None] * N * r
        req_c = [None] * N

        if together:
            for i in range(N):
                for j in range(r):
                    req_a[i + j * N] = communicators.comm.Isend([Aenc[i][j], MPI.INT], dest=i + 1, tag=15)
                    req_b[i + j * N] = communicators.comm.Isend([Benc[i][j], MPI.INT], dest=i + 1, tag=29)
                req_c[i] = communicators.comm.Irecv([Crtn[i], MPI.INT], source=i + 1, tag=42)

            MPI.Request.Waitall(req_a)
            MPI.Request.Waitall(req_b)

            if barrier:
                communicators.comm.Barrier()

            MPI.Request.Waitall(req_c)

        else:

            req_ab = [None] * 2 * N * r

            for i in range(N):
                for j in range(r):
                    communicators.comm.Isend([Aenc[i][j], MPI.INT], dest=i + 1, tag=15)
                    req_ab[i + (j + r) * N] = communicators.comm.Isend([Benc[i][j], MPI.INT], dest=i + 1, tag=29)
                req_c[i] = communicators.comm.Irecv([Crtn[i], MPI.INT], source=i + 1, tag=42)

            if barrier:
                communicators.comm.Barrier()

            for i in range(N):
                j = MPI.Request.Waitany(req_c)
                tmp = time.time()

        res = decode_message(d_cross, Crtn, field)

        final_res = res[0:r]

        if barrier:
            communicators.comm.Barrier()

        for i in range(N):
            serv_comp[i] = communicators.comm.recv(source=i + 1, tag=64)

        if verific:
            Cver = []
            for bb in Bn:
                Cver += [(aa * bb.getT()) % field for aa in An]

            print ([np.array_equal(final_res[i], Cver[i]) for i in range(len(Cver))])

        if barrier:
            communicators.comm.Barrier()

        return tools_for_enc_time, enc_time, serv_comp, random


def scs_sl(N, r, field, barrier, m, n, p, flazhok, base):
    if 0 < communicators.prev_comm.rank < N + 1:
        Ai = []
        Bi = []
        recv_a = [None] * r
        recv_b = [None] * r
        if flazhok:
            if r == 1:
                Ci = np.matrix([[0] * (p / r) for i in range(m)])

            else:
                if m <= p:
                    Ci = np.matrix([[0] * (p / r) for i in range(m)])
                else:
                    Ci = np.matrix([[0] * (p) for i in range(m / r)])
        else:
            if r == 1:
                Ci = np.matrix([[0] * (p) for i in range(m / r)])

            else:
                if m <= p:
                    Ci = np.matrix([[0] * (p) for i in range(m / r)])
                else:
                    Ci = np.matrix([[0] * (p / r) for i in range(m)])


        for j in range(r):
            if flazhok:
                if r == 1:
                    Aij = np.empty_like(np.matrix([[0] * n for i in range(m)]))
                    Bij = np.empty_like(np.matrix([[0] * n for i in range(p / r)]))
                else:
                    if m <= p:
                        Aij = np.empty_like(np.matrix([[0] * n for i in range(m)]))
                        Bij = np.empty_like(np.matrix([[0] * n for i in range(p / r)]))
                    else:
                        Aij = np.empty_like(np.matrix([[0] * n for i in range(m / r)]))
                        Bij = np.empty_like(np.matrix([[0] * n for i in range(p)]))
            else:
                if r == 1:
                    Aij = np.empty_like(np.matrix([[0] * n for i in range(m / r)]))
                    Bij = np.empty_like(np.matrix([[0] * n for i in range(p)]))
                else:
                    if m <= p:
                        Aij = np.empty_like(np.matrix([[0] * n for i in range(m / r)]))
                        Bij = np.empty_like(np.matrix([[0] * n for i in range(p)]))
                    else:
                        Aij = np.empty_like(np.matrix([[0] * n for i in range(m)]))
                        Bij = np.empty_like(np.matrix([[0] * n for i in range(p / r)]))


            recv_a[j] = communicators.comm.Irecv(Aij, source=0, tag=15)
            recv_b[j] = communicators.comm.Irecv(Bij, source=0, tag=29)
            Ai.append(Aij)
            Bi.append(Bij)

        MPI.Request.Waitall(recv_a)
        MPI.Request.Waitall(recv_b)

        servcomp_start = time.time()

        for j in range(r):
            Ci += strassen(Ai[j], (Bi[j].getT()), base)
            Ci = Ci % field

  #      for j in range(r):
  #          Ci += Ai[j] * (Bi[j].getT())
  #          Ci = Ci % field

        servcomp_done = time.time()

        servcomp = servcomp_done - servcomp_start

        if barrier:
            communicators.comm.Barrier()

        req_c = communicators.comm.Isend(Ci, dest=0, tag=42)
        req_c.Wait()

        if barrier:
            communicators.comm.Barrier()

        communicators.comm.send(servcomp, dest=0, tag=64)
        communicators.comm.send(servcomp_start, dest=0, tag=70)

        if barrier:
            communicators.comm.Barrier()
