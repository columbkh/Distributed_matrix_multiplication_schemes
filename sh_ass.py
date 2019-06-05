from tools import *
from mpi4py import MPI
import time
import communicators


def ass_m(N, l, r_a, r_b, k, rt, F, barrier, verific, together, A, B, m, n, p, q):
    if communicators.prev_comm.rank == 0:
        Ap = np.split(A, r_a)
        Bp = np.split(B, r_b)
        Ka = [np.matrix(np.random.random_integers(0, 255, (m / r_a, n))) for i in range(l)]
        Kb = [np.matrix(np.random.random_integers(0, 255, (p / r_b, n))) for i in range(l)]

        x = [pow(rt, i, F) for i in range(k)]
        t = 3
        for i in range(N - k):
            while is_power2(t):
                t += 1
            x.append(t)

        Aenc = getAenc(Ap, Ka, N, F, l, r_a, x)
        Benc = getBenc(Bp, Kb, N, F, l, r_a, r_b, x)

        Rdict = []
        for i in range(N):
            Rdict.append(np.zeros((m / r_a, p / r_b), dtype=np.int_))
        Crtn = []
        serv_comp = [None] * N
        ul_stop = [None] * N
        for i in range(N):
            Crtn.append(np.zeros((m / r_a, p / r_b), dtype=np.int_))

        reqA = [None] * N
        reqB = [None] * N
        reqC = [None] * N
        reqS = [None] * N

        if together:
            ul_start = time.time()
            for i in range(N):
                reqA[i] = communicators.comm.Isend([Aenc[i], MPI.INT], dest=i + 1, tag=15)
                reqB[i] = communicators.comm.Isend([Benc[i], MPI.INT], dest=i + 1, tag=29)
                reqC[i] = communicators.comm.Irecv([Rdict[i], MPI.INT], source=i + 1, tag=42)

            MPI.Request.Waitall(reqA)
            MPI.Request.Waitall(reqB)

            if barrier:
                communicators.comm.Barrier()

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
            ul_start = [None] * N

            dl = [None] * N
            reqAB = [None] * 2 * N
            for i in range(N):
                ul_start[i] = time.time()
                reqAB[i] = communicators.comm.Isend([Aenc[i], MPI.INT], dest=i + 1, tag=15)
                reqAB[i + N] = communicators.comm.Isend([Benc[i], MPI.INT], dest=i + 1, tag=29)
                reqC[i] = communicators.comm.Irecv([Rdict[i], MPI.INT], source=i + 1, tag=42)

            lsst = []
            lst = []

            for i in range(N * 2):
                j = MPI.Request.Waitany(reqAB)

            if barrier:
                communicators.comm.Barrier()

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

        dec_start = time.time()

        interpol(missing, Crtn, F, k, lst, x)
        inv_matr = get_dec_matr(x[:k], F)
        res = decode_message(inv_matr, Crtn[:k], F)

        final_res = []

        for k_tmp in range(r_b):
            final_res += res[k_tmp * (r_a + l): (k_tmp + 1) * (r_a + l) - l]

        dec_done = time.time()
        dec = dec_done - dec_start

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
            for bb in Bp:
                Cver += [(aa * bb.getT()) % F for aa in Ap]

            print ([np.array_equal(final_res[i], Cver[i]) for i in range(len(Cver))])

        if barrier:
            communicators.comm.Barrier()

        return dec, dl, ul, serv_comp


def ass_sl(N, l, r_a, r_b, k, rt, F, barrier, verific, together, m, n, p, q):
    if 0 < communicators.prev_comm.rank < N + 1:
        Ai = np.empty_like(np.matrix([[0] * n for i in range(m / r_a)]))
        Bi = np.empty_like(np.matrix([[0] * n for i in range(p / r_b)]))
        rA = communicators.comm.Irecv(Ai, source=0, tag=15)
        rB = communicators.comm.Irecv(Bi, source=0, tag=29)

        rA.wait()
        rB.wait()

        servcomp_start = time.time()

        Ci = (Ai * (Bi.getT())) % F

        servcomp_done = time.time()

        servcomp = servcomp_done - servcomp_start

        if barrier:
            communicators.comm.Barrier()

        sC = communicators.comm.Isend(Ci, dest=0, tag=42)
        sC.Wait()

        if barrier:
            communicators.comm.Barrier()

        sSrv = communicators.comm.send(servcomp, dest=0, tag=64)
        sUpl = communicators.comm.send(servcomp_start, dest=0, tag=70)

        if barrier:
            communicators.comm.Barrier()
