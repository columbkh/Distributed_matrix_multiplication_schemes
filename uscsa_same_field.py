from gasp_sh import *
from sh_ass import *
from sh_scs import *
from scs_win import *
from sh_uscsa import *
from sh_gscsa import *
from sh_ass_so import *
import numpy as np
import argparse


def do_scs(N, l, r, field, barrier, verific, together, A, B, m, n, p, i, scs, flazhok):
    if MPI.COMM_WORLD.rank == 0:
        if flazhok:
            print "scs winograd: Iteration", str(i)
            enc, dec, dl, ul, comp = win_scs_m(N, l, r, field, barrier, verific, together, A, B, m, p, True)
            compute(enc, dec, dl, ul, comp, scs, i)
        else:
            print "scs: Iteration", str(i)
            enc, dec, dl, ul, comp = scs_m(N, l, r, field, barrier, verific, together, A, B, m, p, True)
            compute(enc, dec, dl, ul, comp, scs, i)
    else:
        if flazhok:
            win_scs_sl(N, r, field, barrier, m, n, p, True)
        else:
            scs_sl(N, r, field, barrier, m, n, p, True)


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


def do_ass(N, l, r_a_ass, r_b_ass, k, rt, field, barrier, verific, together, A, B, m, n, p, i, ass, flazhok):
    if MPI.COMM_WORLD.rank == 0:
        if flazhok:
            print "ass: Iteration", str(i)
            enc, dec, dl, ul, comp = ass_m(N, l, r_a_ass, r_b_ass, k, rt, field, barrier, verific, together, A, B, m, n, p)
            compute(enc, dec, dl, ul, comp, ass, i)
        else:
            print "ass second order: Iteration", str(i)
            enc, dec, dl, ul, comp = ass_m_so(N, l, r_a_ass, r_b_ass, k, rt, field, barrier, verific, together, A, B, m, n, p)
            compute(enc, dec, dl, ul, comp, ass, i)

    else:
        if flazhok:
            ass_sl(N, r_a_ass, r_b_ass, field, barrier, m, n, p)
        else:
            ass_sl_so(N, r_a_ass, r_b_ass, field, barrier, m, n, p)


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


def do_test(r_a, r_b, l, field, q, m, n, p, verific, together):
    A = None
    B = None
    gasp = None
    ass = None
    ass_so = None
    scs = None
    scs_ruck = None
    uscsa_ruck = None
    uscsa = None
    gscsa = None
    gscsa_ruck = None

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

  #  qq, f = factorize_root(N + 1 - 2 * l)
  #  qq = qq - 1
    #qq = 3
    #f = 2

    qq = 4
    f = 3




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
    tmp = lcm(r, r_b_ass)
    tmp_p = lcm(tmp, r_a_ass)
    lcm_m = lcm(dev, tmp_m)
    lcm_p = lcm(dev, tmp_p)

    if m % lcm_m != 0:
        m = (m // lcm_m) * lcm_m
    if p % lcm_p != 0:
        p = (p // lcm_p) * lcm_p

#    if m % r_a_ass != 0:
#        m = (m // r_a_ass) * r_a_ass
#    if p % tmp_p != 0:
#        p = (p // tmp_p) * tmp_p

    #if p % r != 0:
     #   p = (p // r) * r


    if MPI.COMM_WORLD.rank == 0:
        print "actual m: ", m
        print "actual p: ", p

        print "r: ", r
        print "r_a: ", r_a_ass
        print "r_b: ", r_b_ass

    if MPI.COMM_WORLD.rank == 0:
        gasp = [np.zeros(q) for count in range(5)]
        ass = [np.zeros(q) for count in range(5)]
        ass_so = [np.zeros(q) for count in range(5)]
        scs = [np.zeros(q) for count in range(5)]
        scs_ruck = [np.zeros(q) for count in range(5)]
        uscsa_ruck = [np.zeros(q) for count in range(5)]
        uscsa = [np.zeros(q) for count in range(5)]
        gscsa = [np.zeros(q) for count in range(5)]
        gscsa_ruck = [np.zeros(q) for count in range(5)]



    for i in range(q):
        if MPI.COMM_WORLD.rank == 0:
            A = np.matrix(np.random.random_integers(0, 255, (m, n)))
            B = np.matrix(np.random.random_integers(0, 255, (p, n)))
        do_gasp(r_a, r_b, l, N, field, True, verific, together, A, B, m, n, p, i, gasp)
        do_ass(N, l, r_a_ass, r_b_ass, k, rt, field, True, verific, together, A, B, m, n, p, i, ass, True)
       # do_ass(N, l, r_a_ass, r_b_ass, k, rt, field, True, verific, together, A, B, m, n, p, i, ass_so, False)
        #do_scs(N, l, r, field, True, verific, together, A, B, m, n, p, i, scs, True)
        do_scs(N, l, r, field, True, verific, together, A, B, m, n, p, i, scs_ruck, False)
        do_uscsa(N, l, f, qq, field, True, verific, together, A, B, m, n, p, i, uscsa, True)
        do_gscsa(N, l, f, qq, field, True, verific, together, A, B, m, n, p, i, gscsa, True)
       # do_gscsa(N, l, f, qq, field, True, verific, together, A, B, m, n, p, i, gscsa_ruck, False)
       # do_uscsa(N, l, f, qq, field, True, verific, together, A, B, m, n, p, i, uscsa_ruck, False)

    if MPI.COMM_WORLD.rank == 0:
        write_to_octave(uscsa, "uscsa" + experiment_name)
        write_to_octave(gscsa, "gscsa" + experiment_name)
        write_to_octave(scs_ruck, "scsa" + experiment_name)
     #   write_to_octave(scs_ruck, "scsa_ruck" + experiment_name)
        write_to_octave(gasp, "gasp" + experiment_name)
        write_to_octave(ass, "ass" + experiment_name)
      #  write_to_octave(ass_so, "ass_so" + experiment_name)
        #write_to_octave(uscsa_ruck, "uscsa_ruck" + experiment_name)
        #write_to_octave(gscsa_ruck, "gscsa_ruck" + experiment_name)


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
