from gasp_sh import *
from gaspr_sh import *
import numpy as np
import argparse

def do_gasp(r_a, r_b, l, N, field, barrier, verific, together, A, B, m, n, p, i, gasp):
    if MPI.COMM_WORLD.rank == 0:
        print "gasp: Iteration", str(i)
        enc, dec, dl, ul, comp = gasp_m(r_a, r_b, l, field, barrier, verific, together, A, B, m, n, p)
        compute(enc, dec, dl, ul, comp, gasp, i)
    else:
        gasp_sl(r_a, r_b, N, field, barrier, m, n, p)


def do_gaspr(r_a, r_b, l, N, r, field, barrier, verific, together, A, B, m, n, p, i, gaspr):
    if MPI.COMM_WORLD.rank == 0:
        print "gasp_r: Iteration", str(i)
        enc, dec, dl, ul, comp = gaspr_m(r_a, r_b, l, field, barrier, verific, together, A, B, m, n, p, r)
        compute(enc, dec, dl, ul, comp, gaspr, i)
    else:
        gaspr_sl(r_a, r_b, N, field, barrier, m, n, p)


def compute(enc, dec, dl, ul, comp, scheme, i):
    print "dl: ", dl
    print "ul: ", ul
    print "dec: ",dec
    print "comp: ", comp
    print "enc: ", enc

    if isinstance(dl, list):
        scheme[0][i] = sum(dl) / len(dl)
    else:
        scheme[0][i] = dl

    if isinstance(ul, list):
        scheme[1][i] = sum(ul) / len(ul)
    else:
        scheme[1][i] = ul

    if isinstance(comp, list):
        scheme[3][i] = sum(comp) / len(comp)
    else:
        scheme[3][i] = comp

    scheme[2][i] = dec
    scheme[4][i] = enc


def do_test(r_a, r_b, l, r, field, q, m, n, p, verific, together):
    A = None
    B = None
    gasp = None
    gaspr = None

    experiment_name = "_Q_" + str(q) + "_m_" + str(m) + "_n_" + str(n) + "_p_" + str(p)


    if l >= min(r_a, r_b):
        inv_matr, an, ter, N, a, b = create_GASP_big(r_a, r_b, l, field)
    else:
        inv_matr, an, ter, N, a, b = create_GASP_small(r_a, r_b, l, field)

    inv_matr, an, ter, NN, a, b = create_GASP_r(r_a, r_b, l, r, field)


    if MPI.COMM_WORLD.rank == 0:
        print "N: ", N
        print "N for GASPr: ", NN
        print "r: ", r
        print "r_a_gasp: ", r_a
        print "r_b_gasp: ", r_b

    if m % r_a != 0:
        m = (m // r_a) * r_a
    if p % r_b != 0:
        p = (p // r_b) * r_b

    if MPI.COMM_WORLD.rank == 0:
        print "actual m: ", m
        print "actual p: ", p

        print "r: ", r

    if MPI.COMM_WORLD.rank == 0:
        gasp = [np.zeros(q) for count in range(5)]
        gaspr = [np.zeros(q) for count in range(5)]

    for i in range(q):
        if MPI.COMM_WORLD.rank == 0:
            A = np.matrix(np.random.random_integers(0, field-1, (m, n)))
            B = np.matrix(np.random.random_integers(0, field-1, (p, n)))
        do_gasp(r_a, r_b, l, N, field, True, verific, together, A, B, m, n, p, i, gasp)
        do_gaspr(r_a, r_b, l, NN, r, field, True, verific, together, A, B, m, n, p, i, gaspr)
    if MPI.COMM_WORLD.rank == 0:
        write_to_octave(gasp, "gasp" + experiment_name)
        write_to_octave(gaspr, "gaspr" + experiment_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--Field', type=int, help='Finite Field')
    parser.add_argument('--r_a', type=int, help='divide A on K')
    parser.add_argument('--r_b', type=int, help='divide B on L')
    parser.add_argument('--l', type=int, help='number of colluding workers')
    # parser.add_argument('--barrier', help='Enable barrier', action="store_true")
    parser.add_argument('--verific', help='Enable Verification', action="store_true")
    parser.add_argument('--all_together', help='Compute all together', action="store_true")
    parser.add_argument('--matr_size', type=int, help='Compute all together')
    parser.add_argument('--Q', type=int, help='Number of iterations')

    args = parser.parse_args()

    if args.verific:
        verific = True
    else:
        verific = False

    if args.all_together:
        together = True
    else:
        together = False

    r_a = args.r_a
    r_b = args.r_b
    l = args.l
    field = args.Field
    q = args.Q

    m = args.matr_size
    n = args.matr_size
    p = args.matr_size

    do_test(r_a, r_b, l, field, q, m, n, p, verific, together)
