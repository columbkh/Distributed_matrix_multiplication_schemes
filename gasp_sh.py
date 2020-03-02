from tools import *
from mpi4py import MPI
import time
import sys
import communicators


def gasp_m(r_a, r_b, l, field, barrier, verific, together, A, B, m, n, p):
    if communicators.prev_comm.rank == 0:
        dec_start = time.time()

        if l >= min(r_a, r_b):
            inv_matr, an, ter, N, a, b = create_GASP_big(r_a, r_b, l, field)
        else:
            inv_matr, an, ter, N, a, b = create_GASP_small(r_a, r_b, l, field)

        dec_pause = time.time()
        dec_firstpart = dec_pause - dec_start

        Ap = np.split(A, r_a)
        Bp = np.split(B, r_b)

        enc_start = time.time()

        Ka = [np.matrix(np.random.random_integers(0, 255, (m / r_a, n))) for i in range(l)]
        Kb = [np.matrix(np.random.random_integers(0, 255, (p / r_b, n))) for i in range(l)]

        Ap += Ka
        Bp += Kb

        Aenc = getAencGASP(Ap, field, N, a, an)
        Benc = getBencGASP(Bp, field, N, b, an)

        enc_stop = time.time()
        enc = enc_stop - enc_start

        if N > 19:
            print "Too many instances"
            sys.exit(100)

        Crtn = []
        serv_comp = [None] * N
        ul_stop = [None] * N
        for i in range(N):
            Crtn.append(np.zeros((m / r_a, p / r_b), dtype=np.int_))

        req_a = [None] * N
        req_b = [None] * N
        req_c = [None] * N

        if together:
            ul_start = time.time()
            for i in range(N):
                req_a[i] = communicators.comm.Isend([Aenc[i], MPI.INT], dest=i + 1, tag=15)
                req_b[i] = communicators.comm.Isend([Benc[i], MPI.INT], dest=i + 1, tag=29)
                req_c[i] = communicators.comm.Irecv([Crtn[i], MPI.INT], source=i + 1, tag=42)

            MPI.Request.Waitall(req_a)
            MPI.Request.Waitall(req_b)

            if barrier:
                communicators.comm.Barrier()

            dl_start = time.time()
            MPI.Request.Waitall(req_c)

            dl_stop = time.time()
            dl = dl_stop - dl_start

        else:
            ul_start = [None] * N
            dl = [None] * N
            req_ab = [None] * 2 * N
            for i in range(N):
                ul_start[i] = time.time()
                req_ab[i] = communicators.comm.Isend([Aenc[i], MPI.INT], dest=i + 1, tag=15)
                req_ab[i + N] = communicators.comm.Isend([Benc[i], MPI.INT], dest=i + 1, tag=29)
                req_c[i] = communicators.comm.Irecv([Crtn[i], MPI.INT], source=i + 1, tag=42)

            if barrier:
                communicators.comm.Barrier()

            dl_start = time.time()

            for i in range(N):
                j = MPI.Request.Waitany(req_c)
                tmp = time.time()
                dl[j] = tmp - dl_start

        dec_pause = time.time()

        res = decode_message(inv_matr, Crtn, field)

        final_res = res[:r_b]

        for k in range(r_a - 1):
            final_res += res[k * (1 + r_b) + r_b + l:(k + 1) * (r_b + 1) + l + r_b - 1]

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
            Cver = []
            for aa in Ap[:r_a]:
                Cver += [(aa * bb.getT()) % field for bb in Bp[:r_b]]

            print ([np.array_equal(final_res[i], Cver[i]) for i in range(len(Cver))])

        if barrier:
            communicators.comm.Barrier()

        return enc, dec, dl, ul, serv_comp


def gasp_sl(r_a, r_b, N, field, barrier, m, n, p):
    if 0 < communicators.prev_comm.rank < N + 1:
        Ai = np.empty_like(np.matrix([[0] * n for i in range(m / r_a)]))
        Bi = np.empty_like(np.matrix([[0] * n for i in range(p / r_b)]))
        recv_a = communicators.comm.Irecv(Ai, source=0, tag=15)
        recv_b = communicators.comm.Irecv(Bi, source=0, tag=29)

        recv_a.wait()
        recv_b.wait()

        servcomp_start = time.time()

        Ci = (Ai * (Bi.getT())) % field

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
