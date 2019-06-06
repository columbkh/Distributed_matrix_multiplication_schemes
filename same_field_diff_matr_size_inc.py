from same_field_choose_ra_rb import *


def set_communicators(r_a, r_b, l, field):
    if l == min(r_a, r_b):
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
            N = possb.N

    communicators.prev_comm = MPI.COMM_WORLD
    if N + 1 < communicators.prev_comm.Get_size():
        instances = [i for i in range(N + 1, communicators.prev_comm.Get_size())]
        new_group = communicators.prev_comm.group.Excl(instances)
        communicators.comm = communicators.prev_comm.Create(new_group)
    else:
        communicators.comm = communicators.prev_comm


def do_multiple_tests(r_a, r_b, l, field, q, m, n, p, verific, together, number, inc):
    write_title_to_octave(q, field, m, number, inc)
    set_communicators(r_a, r_b, l, field)

    for i in range(number):
        if MPI.COMM_WORLD.rank == 0:
            print("--------------------------------------------------------------------------")
            print("Experiment Nr. ", i + 1)
            print("m: ", m)
            print("n: ", n)
            print("p: ", p)
            print("--------------------------------------------------------------------------")
        do_test(r_a, r_b, l, field, q, m, n, p, verific, together)
        m = m + inc
        p = p + inc
        n = n + inc


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
    parser.add_argument('--Inc', type=int, help='Increment')

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

    m = args.start_matr_size
    n = args.start_matr_size
    p = args.start_matr_size

    number = args.Number
    inc = args.Inc

    do_multiple_tests(r_a, r_b, l, field, q, m, n, p, verific, together, number, inc)
