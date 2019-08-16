from tools import *
from mpi4py import *
import time
import sys
import communicators


def uscsa_m(N, l, f, q, field, barrier, verific, together, A, B, m, p):
    if communicators.prev_comm.rank == 0:
        if N > 19:
            print "Too many instances"
            sys.exit(100)

        Bn = np.split(B, q)
        An = np.split(A, f)

        dec_start = time.time()

        d_cross, left_part, j_plus_i_plus_an, i_plus_an, an = uscsa_make_matrix_d_cross(N, field, q, f, l)
        dec_pause = time.time()
        dec_firstpart = dec_pause - dec_start

        Aenc = uscsa_encode_A(left_part, i_plus_an, An, field, N, l, f, q)
        Benc = uscsa_encode_B(Bn, i_plus_an, field, l, q, f, N, j_plus_i_plus_an)

        Crtn = []
        serv_comp = [None] * N
        ul_stop = [None] * N

        for i in range(N):
            Crtn.append(np.zeros((m / f, p / q), dtype=np.int_))

        req_a = [None] * N * q
        req_b = [None] * N * q
        req_c = [None] * N

        if together:
            ul_start = time.time()
            for i in range(N):
                for j in range(q):
                    req_a[i + j * N] = communicators.comm.Isend([Aenc[i][j], MPI.INT], dest=i + 1, tag=15)
                    req_b[i + j * N] = communicators.comm.Isend([Benc[i][j], MPI.INT], dest=i + 1, tag=29)
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
            req_ab = [None] * 2 * N * q

            for i in range(N):
                ul_start[i] = time.time()
                for j in range(q):
                    communicators.comm.Isend([Aenc[i][j], MPI.INT], dest=i + 1, tag=15)
                    req_ab[i + (j + q) * N] = communicators.comm.Isend([Benc[i][j], MPI.INT], dest=i + 1, tag=29)
                req_c[i] = communicators.comm.Irecv([Crtn[i], MPI.INT], source=i + 1, tag=42)

            if barrier:
                communicators.comm.Barrier()

            dl_start = time.time()

            for i in range(N):
                j = MPI.Request.Waitany(req_c)
                tmp = time.time()
                dl[j] = tmp - dl_start

        dec_pause = time.time()

        res = decode_message(d_cross, Crtn, field)

        final_res = res[0:q*f]

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
            for bb in Bn:
                Cver += [(aa * bb.getT()) % field for aa in An]

            print ([np.array_equal(final_res[i], Cver[i]) for i in range(len(Cver))])

        if barrier:
                communicators.comm.Barrier()

        return dec, dl, ul, serv_comp


def uscsa_sl(N, q, f, field, barrier, m, n, p):
        if 0 < communicators.prev_comm.rank < N + 1:
            Ai = []
            Bi = []
            recv_a = [None] * q
            recv_b = [None] * q
            Ci = np.matrix([[0] * (p / q) for i in range(m / f)])
            for j in range(q):
                Aij = np.empty_like(np.matrix([[0] * n for i in range(m / f)]))
                Bij = np.empty_like(np.matrix([[0] * n for i in range(p / q)]))
                recv_a[j] = communicators.comm.Irecv(Aij, source=0, tag=15)
                recv_b[j] = communicators.comm.Irecv(Bij, source=0, tag=29)
                Ai.append(Aij)
                Bi.append(Bij)

            MPI.Request.Waitall(recv_a)
            MPI.Request.Waitall(recv_b)

            servcomp_start = time.time()

            for j in range(q):
                Ci += (Ai[j] * (Bi[j].getT()))
                Ci = Ci % field

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