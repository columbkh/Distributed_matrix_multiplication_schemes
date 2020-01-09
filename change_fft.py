from a3s_vs_a3sfft import *


def do_multiple_tests(N, l, field, q, m, n, p, verific, together, number, coeff):
    write_title_to_octavemnp(q, field, m, n, p, number, coeff)
    set_communicatorsNl(N, l, field)
    set_communicatorsNlgasp(N, l, field)

    for i in range(number):
        if MPI.COMM_WORLD.rank == 0:
            print "--------------------------------------------------------------------------"
            print "Experiment Nr. ", i + 1
            print "m: ", m
            print "n: ", n
            print "p: ", p
            print "--------------------------------------------------------------------------"
        do_test(N, l, field, q, m, n, p, verific, together)
        m = get_rounded(m * coeff)
        p = get_rounded(p * coeff)
        n = get_rounded(n * coeff)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--Field', type=int, help='Finite Field')
    parser.add_argument('--N', type=int, help='Number of workers')
    parser.add_argument('--l', type=int, help='number of colluding workers')
    parser.add_argument('--verific', help='Enable Verification', action="store_true")
    parser.add_argument('--all_together', help='Compute all together', action="store_true")
    parser.add_argument('--Q', type=int, help='Number of iterations')
    parser.add_argument('--Number', type=int, help='Number of experiments')
    parser.add_argument('--Coeff', type=float, help='Coefficient')
    parser.add_argument('--m', type=int, help='Starting matr size m')
    parser.add_argument('--n', type=int, help='Starting matr size n')
    parser.add_argument('--p', type=int, help='Starting matr size p')
    args = parser.parse_args()

    if args.verific:
        verific = True
    else:
        verific = False

    if args.all_together:
        together = True
    else:
        together = False

    N = args.N
    l = args.l
    field = args.Field
    q = args.Q

    m = args.m
    n = args.n
    p = args.p

    number = args.Number
    coeff = args.Coeff

    do_multiple_tests(N, l, field, q, m, n, p, verific, together, number, coeff)

