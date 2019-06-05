from same_field_choose_ra_rb import *


def do_multiple_tests(r_a, r_b, l, Field, Q, m, n, p, verific, together, Number, coeff):
    write_title_to_octave(Q, Field, m, Number, coeff)
    set_communicators(r_a, r_b, l, Field)
    for i in range(Number):
        if MPI.COMM_WORLD.rank == 0:
            print "--------------------------------------------------------------------------"
            print "Experiment Nr. ", i + 1
            print "m: ", m
            print "n: ", n
            print "p: ", p
            print "--------------------------------------------------------------------------"
        do_test(r_a, r_b, l, Field, Q, m, n, p, verific, together)
        m = get_rounded(m * coeff)
        p = get_rounded(p * coeff)
        n = get_rounded(n * coeff)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--Field', type=int, help='Finite Field')
    parser.add_argument('--r_a', type=int, help='divide A on K')
    parser.add_argument('--r_b', type=int, help='divide B on L')
    parser.add_argument('--l', type=int, help='number of colluding workers')
    parser.add_argument('--verific', help='Enable Verification', action="store_true")
    parser.add_argument('--all_together', help='Compute all together', action="store_true")
    parser.add_argument('--start_matr_size', type=int, help='Starting matr size')
    parser.add_argument('--Q', type=int, help='Number of iterations')
    parser.add_argument('--Number', type=int, help='Number of experiments')
    parser.add_argument('--Coeff', type=float, help='Coefficient')

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
    Field = args.Field
    Q = args.Q

    m = args.start_matr_size
    n = args.start_matr_size
    p = args.start_matr_size

    Number = args.Number
    coeff = args.Coeff

    do_multiple_tests(r_a, r_b, l, Field, Q, m, n, p, verific, together, Number, coeff)
