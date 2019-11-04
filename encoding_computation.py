from ec_scsa import *
from ec_uscsa import *
from ec_gscsa import *
import numpy as np
import argparse


def do_scs(N, l, r, field, barrier, verific, together, A, B, m, n, p, i, scs, flazhok, base):
    if MPI.COMM_WORLD.rank == 0:
        print "scs: Iteration", str(i)
        tools, enc, comp, random = scs_m(N, l, r, field, barrier, verific, together, A, B, m, p, flazhok, i)
        compute(tools, enc, comp, random, scs, i)
    else:
        scs_sl(N, r, field, barrier, m, n, p, flazhok, base)


def do_uscsa(N, l, f, qq, field, barrier, verific, together, A, B, m, n, p, i, uscsa, flazhok):
    if MPI.COMM_WORLD.rank == 0:
        print "uscsa: Iteration", str(i)
        tools, enc, comp = uscsa_m(N, l, f, qq, field, barrier, verific, together, A, B, m, p, flazhok)
        compute(tools, enc, comp, uscsa, i)
    else:
        uscsa_sl(N, qq, f, field, barrier, m, n, p, flazhok)


def do_gscsa(N, l, f, qq, field, barrier, verific, together, A, B, m, n, p, i, gscsa, flazhok):
    if MPI.COMM_WORLD.rank == 0:
        print "gscsa: Iteration", str(i)
        tools, enc, comp = gscsa_m(N, l, f, qq, field, barrier, verific, together, A, B, m, p, flazhok)
        compute(tools, enc, comp, gscsa, i)
    else:
        gscsa_sl(N, qq, f, field, barrier, m, n, p, flazhok)


def compute(tools, enc, comp, random, scheme, i):
    if isinstance(comp, list):
        scheme[2][i] = sum(comp) / len(comp)
    else:
        scheme[2][i] = comp

    scheme[0][i] = tools
    scheme[1][i] = enc
    scheme[3][i] = random


def do_test(r_a, r_b, l, field, q, m, n, p, verific, together):
    A = None
    B = None
    scs = None

    experiment_name = "_Q_" + str(q) + "_m_" + str(m) + "_n_" + str(n) + "_p_" + str(p)

    if l >= min(r_a, r_b):
        inv_matr, an, ter, N, a, b = create_GASP_big(r_a, r_b, l, field)
    else:
        inv_matr, an, ter, N, a, b = create_GASP_small(r_a, r_b, l, field)

    if not is_prime_number(field):
        print "Field is not prime"
        sys.exit(100)
    else:
        possb = get_nofft_for_fixedN(N, l)
        if not possb:
            print "No possabilities"
            sys.exit(100)
        else:
            r_a_ass = possb.r_a
            r_b_ass = possb.r_b
            N = possb.N
            l = possb.l
            k = possb.k
            rt = find_var(field, k)

    r = N - 2 * l

    #qq = 3
    #f = 2

    qq = 4
    f = 3
    ord = 4

    if MPI.COMM_WORLD.rank == 0:
        print "N: ", N
        print "r: ", r
        print "r_a_gasp: ", r_a
        print "r_b_gasp: ", r_b
        print "r_a_ass:", r_a_ass
        print "r_b_ass:", r_b_ass
        print "q: ", qq
        print "f: ", f

    tmp = lcm(r, qq)
    dev = lcm(tmp, f)

    tmp_m = lcm(r_a, r_a_ass)
    tmp_p = lcm(r_b, r_b_ass)
    lcm_m = lcm(dev, tmp_m)
    lcm_p = lcm(dev, tmp_p)
    lcm_m = lcm(lcm_m, r_b_ass)
    lcm_p = lcm(lcm_p, r_a_ass)

    lcm_m = lcm(lcm_m, r_b)
  #  print "lcm_m = ", lcm_m

    if m % lcm_m != 0:
        m = (m // lcm_m) * lcm_m
    if p % lcm_m != 0:
        p = (p // lcm_m) * lcm_m

    tmp = min([m, p])
 #   print "tmp = ", tmp
    base = tmp / r
    m = tmp / r
    p = tmp
    n = tmp / r

    m = m * 2**ord
    p = p * 2**ord
    n = n * 2**ord

    if MPI.COMM_WORLD.rank == 0:
        print "actual m: ", m
        print "actual p: ", p
        print "actual n: ", n
        print "r: ", r

    if MPI.COMM_WORLD.rank == 0:
        scs = [np.zeros(q) for count in range(4)]
        uscsa = [np.zeros(q) for count in range(3)]
        gscsa = [np.zeros(q) for count in range(3)]



    for i in range(q):
        if MPI.COMM_WORLD.rank == 0:
            A = np.matrix(np.random.random_integers(0, 255, (m, n)))
            B = np.matrix(np.random.random_integers(0, 255, (p, n)))
        do_scs(N, l, r, field, True, verific, together, A, B, m, n, p, i, scs, True, base)
     #   do_uscsa(N, l, f, qq, field, True, verific, together, A, B, m, n, p, i, uscsa, True)
     #   do_gscsa(N, l, f, qq, field, True, verific, together, A, B, m, n, p, i, gscsa, True)


    if MPI.COMM_WORLD.rank == 0:
    #    write_to_octave(uscsa, "uscsa" + experiment_name)
    #    write_to_octave(gscsa, "gscsa" + experiment_name)
        write_to_octave(scs, "scsa" + experiment_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--Field', type=int, help='Finite Field')
    parser.add_argument('--r_a', type=int, help='divide A on K')
    parser.add_argument('--r_b', type=int, help='divide B on L')
    parser.add_argument('--l', type=int, help='number of colluding workers')
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
