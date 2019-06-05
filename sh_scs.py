from tools import *
from mpi4py import *
import time
import sys
import communicators


def scs_m(N, l, r, Field, barrier, verific, together, A, B, m, p):
    if communicators.prev_comm.rank == 0:  # Master

        if N > 19:
            print "Too many instances"
            sys.exit(100)

        Bn = np.split(B, r)

        dec_start = time.time()

        d_cross, left_part, i_plus_an, an = make_matrix_d_cross(N, Field, r, l)
        dec_pause = time.time()
        dec_firstpart = dec_pause - dec_start

        Aenc = encode_A(left_part, i_plus_an, A, Field, N, l, r)
        Benc = encode_B(Bn, i_plus_an, Field, l, r, N)

        Crtn = []
        serv_comp = [None] * N
        ul_stop = [None] * N

        for i in range(N):
            Crtn.append(np.zeros((m, p / r), dtype=np.int_))

        reqA = [None] * N * r
        reqB = [None] * N * r
        reqC = [None] * N

        if together:
            ul_start = time.time()
            for i in range(N):
                for j in range(r):
                    reqA[i + j * N] = communicators.comm.Isend([Aenc[i][j], MPI.INT], dest=i + 1, tag=15)
                    reqB[i + j * N] = communicators.comm.Isend([Benc[i][j], MPI.INT], dest=i + 1, tag=29)
                reqC[i] = communicators.comm.Irecv([Crtn[i], MPI.INT], source=i + 1, tag=42)

            MPI.Request.Waitall(reqA)
            MPI.Request.Waitall(reqB)

            if barrier:
                communicators.comm.Barrier()

            dl_start = time.time()
            MPI.Request.Waitall(reqC)

            dl_stop = time.time()
            dl = dl_stop - dl_start

        else:
            ul_start = [None] * N

            dl = [None] * N
            reqAB = [None] * 2 * N * r

            for i in range(N):
                ul_start[i] = time.time()
                for j in range(r):
                    communicators.comm.Isend([Aenc[i][j], MPI.INT], dest=i + 1, tag=15)
                    reqAB[i + (j + r) * N] = communicators.comm.Isend([Benc[i][j], MPI.INT], dest=i + 1, tag=29)
                reqC[i] = communicators.comm.Irecv([Crtn[i], MPI.INT], source=i + 1, tag=42)

            if barrier:
                communicators.comm.Barrier()

            dl_start = time.time()

            for i in range(N):
                j = MPI.Request.Waitany(reqC)
                tmp = time.time()
                dl[j] = tmp - dl_start

        dec_pause = time.time()

        res = decode_message(d_cross, Crtn, Field)

        final_res = res[0:r]

        dec_done = time.time()

        dec_secondpart = dec_done - dec_pause
        dec = dec_firstpart + dec_secondpart

        if barrier:
            communicators.comm.Barrier()

        for i in range(N):
            serv_comp[i] = communicators.comm.recv(source=i + 1, tag=64)
        for i in range(N):
            ul_stop[i] = communicators.comm.recv(source=i + 1, tag=70)

        if together:
            ul_stop_latest = max(ul_stop)
            ul = ul_stop_latest - ul_start
        else:
            ul = [None] * N
            for i in range(N):
                ul[i] = ul_stop[i] - ul_start[i]

        if verific:
            Cver = [(A * bb.getT()) % Field for bb in Bn[:r]]
            print ([np.array_equal(final_res[i], Cver[i]) for i in range(len(Cver))])

        if barrier:
            communicators.comm.Barrier()

        return dec, dl, ul, serv_comp


def scs_sl(N, r, Field, barrier, m, n, p):
    if 0 < communicators.prev_comm.rank < N + 1:
        Ai = []
        Bi = []
        rA = [None] * r
        rB = [None] * r
        Ci = np.matrix([[0] * (p / r) for i in range(m)])
        for j in range(r):
            Aij = np.empty_like(np.matrix([[0] * n for i in range(m)]))
            Bij = np.empty_like(np.matrix([[0] * n for i in range(p / r)]))
            rA[j] = communicators.comm.Irecv(Aij, source=0, tag=15)
            rB[j] = communicators.comm.Irecv(Bij, source=0, tag=29)
            Ai.append(Aij)
            Bi.append(Bij)

        MPI.Request.Waitall(rA)
        MPI.Request.Waitall(rB)

        servcomp_start = time.time()

        for j in range(r):
            Ci += (Ai[j] * (Bi[j].getT()))
            Ci = Ci % Field

        servcomp_done = time.time()

        servcomp = servcomp_done - servcomp_start

        if barrier:
            communicators.comm.Barrier()

        sC = communicators.comm.Isend(Ci, dest=0, tag=42)
        sC.Wait()

        if barrier:
            communicators.comm.Barrier()

        communicators.comm.send(servcomp, dest=0, tag=64)
        communicators.comm.send(servcomp_start, dest=0, tag=70)

        if barrier:
            communicators.comm.Barrier()
