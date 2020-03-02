from tools import *



if __name__ == "__main__":
    field = 53

    print "Polynom Evaluations"

    alphas = [34, 12, 67]
    A = np.matrix(np.random.random_integers(0, field-1, (1200, 1000)))
    An = np.split(A, 30)

    st = []
    st_inv = []
    hs = []
    hs_inv = []
    so = []
    so_inv = []

    start = time.time()
    for al in alphas:
        st.append(standard_eval(An, al, field))
        st_inv.append(standard_eval(An, field-al, field))
    stop = time.time()
    start_hs = time.time()
    for al in alphas:
        hs.append(horner_scheme(An, al, field))
        hs_inv.append(horner_scheme(An, field-al, field))
    stop_hs = time.time()
    start_so = time.time()
    for al in alphas:
        one, another = second_order(An, al, field)
        so.append(one)
        so_inv.append(another)
    stop_so = time.time()
    print [np.equal(st, hs) for i in range(len(st))]
    print [np.equal(st, so) for i in range(len(st))]
    print [np.equal(st_inv, hs_inv) for i in range(len(st_inv))]
    print [np.equal(st_inv, so_inv) for i in range(len(st_inv))]

 #   print "Standard, values: ", st, st_inv
 #   print "Horner, values: ", hs, hs_inv
    print "SO, values: ", so, so_inv

    print "Standard Evaluation: ", stop - start
    print "Horner Scheme: ", stop_hs - start_hs
    print "Second Order: ", stop_so - start_so
