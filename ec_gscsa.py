from tools import *
from mpi4py import *
import time
import sys
import communicators
import argparse

def schema1(A, B, q, f, field, left_part, i_plus_an, N, l, j_plus_i_plus_an, delta):
    An = np.split(A, f*q)
    Bn = np.split(B, 1)

    enc_start = time.time()

    Aenc = gscsa_encode_A(left_part, i_plus_an, An, field, N, l, f, q, delta)
    Benc = gscsa_encode_B(Bn, i_plus_an, field, l, q, f, N, j_plus_i_plus_an)

    enc_stop = time.time()

    return An, Bn, Aenc, Benc, enc_stop - enc_start

def schema2(A, B, q, f, field, left_part, i_plus_an, N, l, j_plus_i_plus_an, delta):
    An = np.split(A, 1)
    Bn = np.split(B, f*q)

    enc_start = time.time()

    Aenc = reverse_gscsa_encode_A(An, i_plus_an, field, l, q, f, N, j_plus_i_plus_an)
    Benc = reverse_gscsa_encode_B(left_part, i_plus_an, Bn, field, N, l, f, q, delta)

    enc_stop = time.time()

    return An, Bn, Aenc, Benc, enc_stop - enc_start

def gscsa_m(N, l, f, q, field, barrier, verific, together, A, B, m, p, flazhok):
    if communicators.prev_comm.rank == 0:
        if N > 19:
            print "Too many instances"
            sys.exit(100)

        tools_for_enc_start = time.time()

        d_cross, left_part, j_plus_i_plus_an, i_plus_an, an, delta = uscsa_make_matrix_d_cross(N, field, q, f, l)

        tools_for_enc_stop = time.time()

        tools_for_enc_time = tools_for_enc_stop - tools_for_enc_start
        enc_time = None

        if flazhok:
            if m == p:
                schema1(A, B, q, f, field, left_part, i_plus_an, N, l, j_plus_i_plus_an, delta)
            else:
                if m > p:
                    if 1 / f <= q:
                        An, Bn, Aenc, Benc, enc_time = schema1(A, B, q, f, field, left_part, i_plus_an, N, l, j_plus_i_plus_an, delta)
                    else:
                        An, Bn, Aenc, Benc, enc_time = schema2(A, B, q, f, field, left_part, i_plus_an, N, l, j_plus_i_plus_an, delta)
                else:
                    if 1 / f >= q:
                        An, Bn, Aenc, Benc, enc_time = schema1(A, B, q, f, field, left_part, i_plus_an, N, l, j_plus_i_plus_an, delta)
                    else:
                        An, Bn, Aenc, Benc, enc_time = schema2(A, B, q, f, field, left_part, i_plus_an, N, l, j_plus_i_plus_an, delta)
        else:
            if m == p:
                An, Bn, Aenc, Benc, enc_time = schema2(A, B, q, f, field, left_part, i_plus_an, N, l, j_plus_i_plus_an, delta)
            else:
                if m > p:
                    if 1 / f <= q:
                        An, Bn, Aenc, Benc, enc_time = schema2(A, B, q, f, field, left_part, i_plus_an, N, l, j_plus_i_plus_an, delta)
                    else:
                        An, Bn, Aenc, Benc, enc_time = schema1(A, B, q, f, field, left_part, i_plus_an, N, l, j_plus_i_plus_an, delta)
                else:
                    if 1 / f >= q:
                        An, Bn, Aenc, Benc, enc_time = schema2(A, B, q, f, field, left_part, i_plus_an, N, l, j_plus_i_plus_an, delta)
                    else:
                        An, Bn, Aenc, Benc, enc_time = schema1(A, B, q, f, field, left_part, i_plus_an, N, l, j_plus_i_plus_an, delta)


        Crtn = []
        serv_comp = [None] * N

        if flazhok:
            if m == p:
                for i in range(N):
                    Crtn.append(np.zeros((m / (f*q), p), dtype=np.int_))
            else:
                if m > p:
                    if 1 / f <= q:
                        for i in range(N):
                            Crtn.append(np.zeros((m / (f*q), p), dtype=np.int_))
                    else:
                        for i in range(N):
                            Crtn.append(np.zeros((m, p / (f*q)), dtype=np.int_))
                else:
                    if 1 / f >= q:
                        for i in range(N):
                            Crtn.append(np.zeros((m / (f*q), p), dtype=np.int_))
                    else:
                        for i in range(N):
                            Crtn.append(np.zeros((m, p / (f*q)), dtype=np.int_))
        else:
            if m == p:
                for i in range(N):
                    Crtn.append(np.zeros((m, p / (f*q)), dtype=np.int_))
            else:
                if m > p:
                    if 1 / f <= q:
                        for i in range(N):
                            Crtn.append(np.zeros((m, p / (f*q)), dtype=np.int_))
                    else:
                        for i in range(N):
                            Crtn.append(np.zeros((m / (f*q), p), dtype=np.int_))
                else:
                    if 1 / f >= q:
                        for i in range(N):
                            Crtn.append(np.zeros((m, p / (f*q)), dtype=np.int_))
                    else:
                        for i in range(N):
                            Crtn.append(np.zeros((m / (f*q), p), dtype=np.int_))

        req_a = [None] * N * q
        req_b = [None] * N * q
        req_c = [None] * N

        if together:
            for i in range(N):
                for j in range(q):
                    req_a[i + j * N] = communicators.comm.Isend([Aenc[i][j], MPI.INT], dest=i + 1, tag=15)
                    req_b[i + j * N] = communicators.comm.Isend([Benc[i][j], MPI.INT], dest=i + 1, tag=29)
                req_c[i] = communicators.comm.Irecv([Crtn[i], MPI.INT], source=i + 1, tag=42)

            MPI.Request.Waitall(req_a)
            MPI.Request.Waitall(req_b)

            if barrier:
                communicators.comm.Barrier()

            MPI.Request.Waitall(req_c)

        else:
            req_ab = [None] * 2 * N * q

            for i in range(N):
                for j in range(q):
                    communicators.comm.Isend([Aenc[i][j], MPI.INT], dest=i + 1, tag=15)
                    req_ab[i + (j + q) * N] = communicators.comm.Isend([Benc[i][j], MPI.INT], dest=i + 1, tag=29)
                req_c[i] = communicators.comm.Irecv([Crtn[i], MPI.INT], source=i + 1, tag=42)

            if barrier:
                communicators.comm.Barrier()

            for i in range(N):
                j = MPI.Request.Waitany(req_c)
                tmp = time.time()
                dl[j] = tmp - dl_start

        res = decode_message(d_cross, Crtn, field)

        final_res = res[0:q*f]

        if barrier:
            communicators.comm.Barrier()

        for i in range(N):
            serv_comp[i] = communicators.comm.recv(source=i + 1, tag=64)


        if verific:
            Cver = []
            if flazhok:
                if m == p:
                    for bb in Bn:
                        Cver += [(aa * bb.getT()) % field for aa in An]
                else:
                    if m > p:
                        if 1 / f <= q:
                            for bb in Bn:
                                Cver += [(aa * bb.getT()) % field for aa in An]
                        else:
                            for aa in An:
                                Cver += [(aa * bb.getT()) % field for bb in Bn]
                    else:
                        if 1 / f >= q:
                            for bb in Bn:
                                Cver += [(aa * bb.getT()) % field for aa in An]
                        else:
                            for aa in An:
                                Cver += [(aa * bb.getT()) % field for bb in Bn]
            else:
                if m == p:
                    for aa in An:
                        Cver += [(aa * bb.getT()) % field for bb in Bn]
                else:
                    if m > p:
                        if 1 / f <= q:
                            for aa in An:
                                Cver += [(aa * bb.getT()) % field for bb in Bn]
                        else:
                            for bb in Bn:
                                Cver += [(aa * bb.getT()) % field for aa in An]
                    else:
                        if 1 / f >= q:
                            for aa in An:
                                Cver += [(aa * bb.getT()) % field for bb in Bn]
                        else:
                            for bb in Bn:
                                Cver += [(aa * bb.getT()) % field for aa in An]

            print ([np.array_equal(final_res[i], Cver[i]) for i in range(len(Cver))])

        if barrier:
            communicators.comm.Barrier()

        return tools_for_enc_time, enc_time, serv_comp

def gscsa_sl(N, q, f, field, barrier, m, n, p, flazhok):
    if 0 < communicators.prev_comm.rank < N + 1:
        Ai = []
        Bi = []
        recv_a = [None] * q
        recv_b = [None] * q
        Ci = np.matrix([[0] * p for i in range(m / (f*q))])

        if flazhok:
            if m == p:
                Ci = np.matrix([[0] * p for i in range(m / (f*q))])
            else:
                if m > p:
                    if 1 / f <= q:
                        Ci = np.matrix([[0] * p for i in range(m / (f * q))])
                    else:
                        Ci = np.matrix([[0] * (p / (f*q)) for i in range(m)])
                else:
                    if 1 / f >= q:
                        Ci = np.matrix([[0] * p for i in range(m / (f * q))])
                    else:
                        Ci = np.matrix([[0] * (p / (f*q)) for i in range(m)])
        else:
            if m == p:
                Ci = np.matrix([[0] * (p / (f*q)) for i in range(m)])
            else:
                if m > p:
                    if 1 / f <= q:
                        Ci = np.matrix([[0] * (p / (f*q)) for i in range(m)])
                    else:
                        Ci = np.matrix([[0] * p for i in range(m / (f * q))])
                else:
                    if 1 / f >= q:
                        Ci = np.matrix([[0] * (p / (f*q)) for i in range(m)])
                    else:
                        Ci = np.matrix([[0] * p for i in range(m / (f * q))])


        for j in range(q):
            Aij = np.empty_like(np.matrix([[0] * n for i in range(m / (f*q))]))
            Bij = np.empty_like(np.matrix([[0] * n for i in range(p)]))
            if flazhok:
                if m == p:
                    Aij = np.empty_like(np.matrix([[0] * n for i in range(m / (f * q))]))
                    Bij = np.empty_like(np.matrix([[0] * n for i in range(p)]))
                else:
                    if m > p:
                        if 1 / f <= q:
                            Aij = np.empty_like(np.matrix([[0] * n for i in range(m / (f * q))]))
                            Bij = np.empty_like(np.matrix([[0] * n for i in range(p)]))
                        else:
                            Aij = np.empty_like(np.matrix([[0] * n for i in range(m)]))
                            Bij = np.empty_like(np.matrix([[0] * n for i in range(p / (f * q))]))
                    else:
                        if 1 / f >= q:
                            Aij = np.empty_like(np.matrix([[0] * n for i in range(m / (f * q))]))
                            Bij = np.empty_like(np.matrix([[0] * n for i in range(p)]))
                        else:
                            Aij = np.empty_like(np.matrix([[0] * n for i in range(m)]))
                            Bij = np.empty_like(np.matrix([[0] * n for i in range(p / (f * q))]))
            else:
                if m == p:
                    Aij = np.empty_like(np.matrix([[0] * n for i in range(m)]))
                    Bij = np.empty_like(np.matrix([[0] * n for i in range(p / (f * q))]))
                else:
                    if m > p:
                        if 1 / f <= q:
                            Aij = np.empty_like(np.matrix([[0] * n for i in range(m)]))
                            Bij = np.empty_like(np.matrix([[0] * n for i in range(p / (f * q))]))
                        else:
                            Aij = np.empty_like(np.matrix([[0] * n for i in range(m / (f * q))]))
                            Bij = np.empty_like(np.matrix([[0] * n for i in range(p)]))
                    else:
                        if 1 / f >= q:
                            Aij = np.empty_like(np.matrix([[0] * n for i in range(m)]))
                            Bij = np.empty_like(np.matrix([[0] * n for i in range(p / (f * q))]))
                        else:
                            Aij = np.empty_like(np.matrix([[0] * n for i in range(m / (f * q))]))
                            Bij = np.empty_like(np.matrix([[0] * n for i in range(p)]))



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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--Field', type=int, help='Finite Field')
    parser.add_argument('--f', type=int, help='divide A on f*q')
    parser.add_argument('--q', type=int, help='divide A on f*q')
    parser.add_argument('--l', type=int, help='number of colluding workers')
    parser.add_argument('--m', type=int, help='Compute all together')
    parser.add_argument('--n', type=int, help='Number of iterations')
    parser.add_argument('--p', type=int, help='Number of iterations')
    parser.add_argument('--N', type=int, help='Number of iterations')

    args = parser.parse_args()

    m = args.m
    n = args.n
    p = args.p

    field = args.Field

    f = args.f
    q = args.q
    l = args.l

    N = args.N

    communicators.prev_comm = MPI.COMM_WORLD
    if N + 1 < communicators.prev_comm.Get_size():
        instances = [i for i in range(N + 1, communicators.prev_comm.Get_size())]
        new_group = communicators.prev_comm.group.Excl(instances)
        communicators.comm = communicators.prev_comm.Create(new_group)
    else:
        communicators.comm = communicators.prev_comm

    if MPI.COMM_WORLD.rank == 0:
        print "gscsa: Iteration"
        A = np.matrix(np.random.random_integers(0, 255, (m, n))) % field
        B = np.matrix(np.random.random_integers(0, 255, (p, n))) % field
        dec, dl, ul, comp = gscsa_m(N, l, f, q, field, True, True, True, A, B, m, p)
    else:
        gscsa_sl(N, q, f, field, True, m, n, p)



