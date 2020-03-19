from gasp_sh import *
from sh_ass import *
from sh_scs import *
from sh_uscsa import *
from sh_gscsa import *
from so_gscsa import *
from sh_ass_so import *
from scs_win import *
from so_uscsa import *
import numpy as np
import argparse


def do_scs(N, l, r, field, barrier, verific, together, A, B, m, n, p, i, scs, flazhok, hs):
    if MPI.COMM_WORLD.rank == 0:
        if flazhok:
            print "scs winograd-second order, horner scheme - ", hs, ": Iteration", str(i)
            enc, dec, dl, ul, comp = win_scs_m(N, l, r, field, barrier, verific, together, A, B, m, p, True, hs)
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


def do_uscsa(N, l, f, qq, field, barrier, verific, together, A, B, m, n, p, i, uscsa, flazhok, so, hs):
    if MPI.COMM_WORLD.rank == 0:
        if so:
            print "uscsa second order, horner scheme - ", hs, ": Iteration", str(i)
            enc, dec, dl, ul, comp = so_uscsa_m(N, l, f, qq, field, barrier, verific, together, A, B, m, p, flazhok, hs)
            compute(enc, dec, dl, ul, comp, uscsa, i)
        else:
            print "uscsa: Iteration", str(i)
            enc, dec, dl, ul, comp = uscsa_m(N, l, f, qq, field, barrier, verific, together, A, B, m, p, flazhok)
            compute(enc, dec, dl, ul, comp, uscsa, i)
    else:
        if so:
            so_uscsa_sl(N, qq, f, field, barrier, m, n, p, flazhok)
        else:
            uscsa_sl(N, qq, f, field, barrier, m, n, p, flazhok)


def do_gscsa(N, l, f, qq, field, barrier, verific, together, A, B, m, n, p, i, gscsa, flazhok, so, hs):
    if MPI.COMM_WORLD.rank == 0:
        if so:
            print "gscsa second order, horner scheme - ", hs, ": Iteration", str(i)
            enc, dec, dl, ul, comp = so_gscsa_m(N, l, f, qq, field, barrier, verific, together, A, B, m, p, flazhok, hs)
            compute(enc, dec, dl, ul, comp, gscsa, i)
        else:
            print "gscsa: Iteration", str(i)
            enc, dec, dl, ul, comp = gscsa_m(N, l, f, qq, field, barrier, verific, together, A, B, m, p, flazhok)
            compute(enc, dec, dl, ul, comp, gscsa, i)
    else:
        if so:
            so_gscsa_sl(N, qq, f, field, barrier, m, n, p, flazhok)
        else:
            gscsa_sl(N, qq, f, field, barrier, m, n, p, flazhok)

def do_gasp(r_a, r_b, l, N, field, barrier, verific, together, A, B, m, n, p, i, gasp):
    if MPI.COMM_WORLD.rank == 0:
        print "gasp: Iteration", str(i)
        enc, dec, dl, ul, comp = gasp_m(r_a, r_b, l, field, barrier, verific, together, A, B, m, n, p)
        compute(enc, dec, dl, ul, comp, gasp, i)
    else:
        gasp_sl(r_a, r_b, N, field, barrier, m, n, p)


def do_ass(N, l, r_a_ass, r_b_ass, k, rt, field, barrier, verific, together, A, B, m, n, p, i, ass, flazhok, hs):
    if MPI.COMM_WORLD.rank == 0:
        if not flazhok:
            print "ass: Iteration", str(i)
            enc, dec, dl, ul, comp = ass_m(N, l, r_a_ass, r_b_ass, k, rt, field, barrier, verific, together, A, B, m, n, p)
            compute(enc, dec, dl, ul, comp, ass, i)
        else:
            print "ass second order, horner scheme - ", hs, ": Iteration", str(i)
            enc, dec, dl, ul, comp = ass_m_so(N, l, r_a_ass, r_b_ass, k, rt, field, barrier, verific, together, A, B, m, n, p, hs)
            compute(enc, dec, dl, ul, comp, ass, i)

    else:
        if flazhok:
            ass_sl(N, r_a_ass, r_b_ass, field, barrier, m, n, p)
        else:
            ass_sl_so(N, r_a_ass, r_b_ass, field, barrier, m, n, p)


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


def do_test(r_a, r_b, l, field, q, m, n, p, qq, f, verific, together):
    A = None
    B = None
    gasp = None
    ass = None
    ass_so = None
    ass_so_hs = None
    scs = None
    scs_win_so = None
    scs_win_so_hs = None
    uscsa = None
    uscsa_so = None
    uscsa_so_hs = None
    gscsa = None
    gscsa_so = None
    gscsa_so_hs = None

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
        ass_so_hs = [np.zeros(q) for count in range(5)]
        scs = [np.zeros(q) for count in range(5)]
        scs_win_so = [np.zeros(q) for count in range(5)]
        scs_win_so_hs = [np.zeros(q) for count in range(5)]
        uscsa = [np.zeros(q) for count in range(5)]
        uscsa_so = [np.zeros(q) for count in range(5)]
        uscsa_so_hs = [np.zeros(q) for count in range(5)]
        gscsa = [np.zeros(q) for count in range(5)]
        gscsa_so = [np.zeros(q) for count in range(5)]
        gscsa_so_hs = [np.zeros(q) for count in range(5)]



    for i in range(q):
        if MPI.COMM_WORLD.rank == 0:
            A = np.matrix(np.random.random_integers(0, field-1, (m, n)))
            B = np.matrix(np.random.random_integers(0, field-1, (p, n)))
        do_gasp(r_a, r_b, l, N, field, True, verific, together, A, B, m, n, p, i, gasp)
        do_ass(N, l, r_a_ass, r_b_ass, k, rt, field, True, verific, together, A, B, m, n, p, i, ass, False, False)
        do_ass(N, l, r_a_ass, r_b_ass, k, rt, field, True, verific, together, A, B, m, n, p, i, ass_so, True, False)
        do_ass(N, l, r_a_ass, r_b_ass, k, rt, field, True, verific, together, A, B, m, n, p, i, ass_so_hs, True, True)
        do_scs(N, l, r, field, True, verific, together, A, B, m, n, p, i, scs, False, False)
        do_scs(N, l, r, field, True, verific, together, A, B, m, n, p, i, scs_win_so, True, False)
        do_scs(N, l, r, field, True, verific, together, A, B, m, n, p, i, scs_win_so_hs, True, True)
        do_uscsa(N, l, f, qq, field, True, verific, together, A, B, m, n, p, i, uscsa, True, False, False)
        do_uscsa(N, l, f, qq, field, True, verific, together, A, B, m, n, p, i, uscsa_so, True, True, False)
        do_uscsa(N, l, f, qq, field, True, verific, together, A, B, m, n, p, i, uscsa_so_hs, True, True, True)
        do_gscsa(N, l, f, qq, field, True, verific, together, A, B, m, n, p, i, gscsa, True, False, False)
        do_gscsa(N, l, f, qq, field, True, verific, together, A, B, m, n, p, i, gscsa_so, True, True, False)
        do_gscsa(N, l, f, qq, field, True, verific, together, A, B, m, n, p, i, gscsa_so_hs, True, True, True)

    if MPI.COMM_WORLD.rank == 0:
        write_to_octave(uscsa, "uscsa" + experiment_name)
        write_to_octave(uscsa_so, "uscsa_so" + experiment_name)
        write_to_octave(uscsa_so_hs, "uscsa_so_hs" + experiment_name)
        write_to_octave(gscsa, "gscsa" + experiment_name)
        write_to_octave(gscsa_so, "gscsa_so" + experiment_name)
        write_to_octave(gscsa_so_hs, "gscsa_so_hs" + experiment_name)
        write_to_octave(scs, "scsa" + experiment_name)
        write_to_octave(scs_win_so, "scsa_win_so" + experiment_name)
        write_to_octave(scs_win_so_hs, "scsa_win_so_hs" + experiment_name)
        write_to_octave(gasp, "gasp" + experiment_name)
        write_to_octave(ass, "ass" + experiment_name)
        write_to_octave(ass_so, "ass_so" + experiment_name)
        write_to_octave(ass_so_hs, "ass_so_hs" + experiment_name)

