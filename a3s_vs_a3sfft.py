from sh_ass import *
from sh_ass_fft import *
import numpy as np
import argparse


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


def do_test(N, l, field, q, m, n, p, verific, together):
    A = None
    B = None
    ass = None
    ass_fft = None

    experiment_name = "_Q_" + str(q) + "_m_" + str(m) + "_n_" + str(n) + "_p_" + str(p)

    if not is_prime_number(field):
        print "Field is not prime"
        sys.exit(100)
    else:
        possb = get_for_fixedNl(N, l)
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
            rts = find_rt(field, k)
            rt_fft = rts[np.random.random_integers(0, len(rts) - 1)]


    if MPI.COMM_WORLD.rank == 0:
        print "r_a_ass:", r_a_ass
        print "r_b_ass:", r_b_ass

    if m % r_a_ass != 0:
        m = (m // r_a_ass) * r_a_ass
    if p % r_b_ass != 0:
        p = (p // r_b_ass) * r_b_ass


    if MPI.COMM_WORLD.rank == 0:
        print "actual m: ", m
        print "actual n: ", n
        print "actual p: ", p

    if MPI.COMM_WORLD.rank == 0:
        ass = [np.zeros(q) for count in range(5)]
        ass_fft = [np.zeros(q) for count in range(5)]



    for i in range(q):
        if MPI.COMM_WORLD.rank == 0:
            A = np.matrix(np.random.random_integers(0, field-1, (m, n)))
            B = np.matrix(np.random.random_integers(0, field-1, (p, n)))
        do_ass(N, l, r_a_ass, r_b_ass, k, rt, field, True, verific, together, A, B, m, n, p, i, ass)
        do_ass_fft(N, l, r_a_ass, r_b_ass, k, rt_fft, field, True, verific, together, A, B, m, n, p, i, ass_fft)



    if MPI.COMM_WORLD.rank == 0:
        write_to_octave(ass, "ass" + experiment_name)
        write_to_octave(ass_fft, "assfft" + experiment_name)