from tools import *
from mpi4py import *
import time
import sys
import communicators

def schema1(A, B, l, r, N, left_part, i_plus_an, field):
    Bn = np.split(B, r)
    shape = A[-1].shape
    field_matr = np.array([[field for el in range(shape[1])] for stroka in range(shape[0])])
    f_x = np.array([[0 for el in range(shape[1])] for stroka in range(shape[0])], dtype='int64')
    g_x = np.array([[0 for el in range(shape[1])] for stroka in range(shape[0])], dtype='int64')
    Aenc = so_encode_A(left_part, i_plus_an, A, field, N, l, r, field_matr, f_x, g_x)
    Aenc_cmp = encode_A(left_part, i_plus_an, A, field, N, l, r)
    Benc = encode_B(Bn, i_plus_an, field, l, r, N)
    return [A], Bn, Aenc, Benc

def schema2(A, B, l, r, N, left_part, i_plus_an, field):
    An = np.split(A, r)
    Aenc = reverse_encode_A(An, i_plus_an, field, l, r, N)
    Benc = reverse_encode_B(left_part, i_plus_an, B, field, N, l, r)
    return An, [B], Aenc, Benc

def win_scs_m(N, l, r, field, barrier, verific, together, A, B, m, p, flazhok):
    if communicators.prev_comm.rank == 0:  # Master

        if N > 19:
            print "Too many instances"
            sys.exit(100)


        dec_start = time.time()

        d_cross, left_part, i_plus_an, an = make_matrix_d_cross_so(N, field, r, l)

        dec_pause = time.time()
        dec_firstpart = dec_pause - dec_start

        enc_start = time.time()

        if flazhok:
            if r == 1:
                An, Bn, Aenc, Benc = schema1(A, B, l, r, N, left_part, i_plus_an, field)
            else:
                if m <= p:
                    An, Bn, Aenc, Benc = schema1(A, B, l, r, N, left_part, i_plus_an, field)
                else:
                    An, Bn, Aenc, Benc = schema2(A, B, l, r, N, left_part, i_plus_an, field)
        else:
            if r == 1:
                An, Bn, Aenc, Benc = schema2(A, B, l, r, N, left_part, i_plus_an, field)
            else:
                if m <= p:
                    An, Bn, Aenc, Benc = schema2(A, B, l, r, N, left_part, i_plus_an, field)
                else:
                    An, Bn, Aenc, Benc = schema1(A, B, l, r, N, left_part, i_plus_an, field)

        enc_stop = time.time()

        enc = enc_stop - enc_start

        Crtn = []
        serv_comp = [None] * N
        ul_stop = [None] * N

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
            ul_start = time.time()
            for i in range(N):
                for j in range(r):
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
            req_ab = [None] * 2 * N * r

            for i in range(N):
                ul_start[i] = time.time()
                for j in range(r):
                    communicators.comm.Isend([Aenc[i][j], MPI.INT], dest=i + 1, tag=15)
                    req_ab[i + (j + r) * N] = communicators.comm.Isend([Benc[i][j], MPI.INT], dest=i + 1, tag=29)
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
            Cver = []
            for bb in Bn:
                Cver += [(aa * bb.getT()) % field for aa in An]

          #  Cver = [(A * bb.getT()) % field for bb in Bn[:r]]
        #    print Cver
        #    print final_res
            print ([np.array_equal(final_res[i], Cver[i]) for i in range(len(Cver))])

        if barrier:
            communicators.comm.Barrier()

        return enc, dec, dl, ul, serv_comp


def win_scs_sl(N, r, field, barrier, m, n, p, flazhok):
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
            Ci += strassen_winograd(Ai[j], Bi[j].getT())
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
