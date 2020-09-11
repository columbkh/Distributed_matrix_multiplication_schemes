from gasp_sh import *
from sh_ass import *
from sh_ass_fft import *
from sh_scs import *
from sh_uscsa import *
from sh_gscsa import *
import numpy as np
import argparse


def do_scs(N, l, r, field, barrier, verific, together, A, B, m, n, p, i, scs, flazhok):
    if MPI.COMM_WORLD.rank == 0:
        print "scs: Iteration", str(i)
        enc, dec, dl, ul, comp = scs_m(N, l, r, field, barrier, verific, together, A, B, m, p, flazhok)
        compute(enc, dec, dl, ul, comp, scs, i)
    else:
        scs_sl(N, r, field, barrier, m, n, p, flazhok)


def do_uscsa(N, l, f, qq, field, barrier, verific, together, A, B, m, n, p, i, uscsa, flazhok):
    if MPI.COMM_WORLD.rank == 0:
        print "uscsa: Iteration", str(i)
        enc, dec, dl, ul, comp = uscsa_m(N, l, f, qq, field, barrier, verific, together, A, B, m, p, flazhok)
        compute(enc, dec, dl, ul, comp, uscsa, i)
    else:
        uscsa_sl(N, qq, f, field, barrier, m, n, p, flazhok)


def do_gscsa(N, l, f, qq, field, barrier, verific, together, A, B, m, n, p, i, gscsa, flazhok):
    if MPI.COMM_WORLD.rank == 0:
        print "gscsa: Iteration", str(i)
        enc, dec, dl, ul, comp = gscsa_m(N, l, f, qq, field, barrier, verific, together, A, B, m, p, flazhok)
        compute(enc, dec, dl, ul, comp, gscsa, i)
    else:
        gscsa_sl(N, qq, f, field, barrier, m, n, p, flazhok)

def do_gasp(r_a, r_b, l, N, field, barrier, verific, together, A, B, m, n, p, i, gasp):
    if MPI.COMM_WORLD.rank == 0:
        print "gasp: Iteration", str(i)
        enc, dec, dl, ul, comp = gasp_m(r_a, r_b, l, field, barrier, verific, together, A, B, m, n, p)
        compute(enc, dec, dl, ul, comp, gasp, i)
    else:
        gasp_sl(r_a, r_b, N, field, barrier, m, n, p)


def do_ass(N, l, r_a_ass, r_b_ass, k, rt, field, barrier, verific, together, A, B, m, n, p, i, ass):
    if MPI.COMM_WORLD.rank == 0:
        print "ass: Iteration", str(i)
        enc, dec, dl, ul, comp = ass_m(N, l, r_a_ass, r_b_ass, k, rt, field, barrier, verific, together, A, B, m, n, p)
        compute(enc, dec, dl, ul, comp, ass, i)
    else:
        ass_sl(N, r_a_ass, r_b_ass, field, barrier, m, n, p)


def do_ass_fft(N, l, r_a_ass, r_b_ass, k, rt, field, barrier, verific, together, A, B, m, n, p, i, ass):
    if MPI.COMM_WORLD.rank == 0:
        print "ass fft: Iteration", str(i)
        enc, dec, dl, ul, comp = ass_fft_m(N, l, r_a_ass, r_b_ass, k, rt, field, barrier, verific, together, A, B, m, n, p)
        compute(enc, dec, dl, ul, comp, ass, i)
    else:
        ass_fft_sl(N, r_a_ass, r_b_ass, field, barrier, m, n, p)


def compute(enc, dec, dl, ul, comp, scheme, i):
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


def do_test(N, ra, rb,  l, field, q, m, n, p, verific, together, k):
    A = None
    B = None

    ass_fft = None
    experiment_name = "_Q_" + str(q) + "_m_" + str(m) + "_n_" + str(n) + "_p_" + str(p)

    rt = find_var(field, k)
    rts = find_rt(field, k)
    rt_fft = rts[np.random.random_integers(0, len(rts) - 1)]







    if MPI.COMM_WORLD.rank == 0:
        print "N: ", N
        print "r_a: ", r_a
        print "r_b: ", r_b


    if MPI.COMM_WORLD.rank == 0:
        print "actual m: ", m
        print "actual p: ", p

    if MPI.COMM_WORLD.rank == 0:
        ass_fft = [np.zeros(q) for count in range(5)]




    for i in range(q):
        if MPI.COMM_WORLD.rank == 0:
            A = np.matrix(np.random.random_integers(0, 255, (m, n)))
            B = np.matrix(np.random.random_integers(0, 255, (p, n)))

        if MPI.COMM_WORLD.rank == 1:
            print N, l, r_a, r_b, k, rt_fft, field, True, verific, together, m, n, p, i
        do_ass_fft(N, l, r_a, r_b, k, rt_fft, field, True, verific, together, A, B, m, n, p, i, ass_fft)


    if MPI.COMM_WORLD.rank == 0:
        write_to_octave(ass_fft, "assfft" + experiment_name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--Field', type=int, help='Finite Field')
    parser.add_argument('--r_a', type=int, help='divide A on K')
    parser.add_argument('--r_b', type=int, help='divide B on L')
    parser.add_argument('--l', type=int, help='number of colluding workers')
    parser.add_argument('--k', type=int, help='k parameter')
    parser.add_argument('--N', type=int, help='N parameter')
    parser.add_argument('--verific', help='Enable Verification', action="store_true")
    parser.add_argument('--all_together', help='Compute all together', action="store_true")
    parser.add_argument('--m', type=int, help='Compute all together')
    parser.add_argument('--n', type=int, help='Compute all together')
    parser.add_argument('--p', type=int, help='Compute all together')
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

    N = args.N

    m = args.m
    n = args.n
    p = args.p

    k = args.k

    set_comms_for_ass_fft(N)

    do_test(N, r_a, r_b, l, field, q, m, n, p, verific, together, k)
