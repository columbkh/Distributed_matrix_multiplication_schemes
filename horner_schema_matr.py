from tools import *

if __name__ == "__main__":
    field = 65537
    if not ffi.field:
        ffi.FF_FIELD = field
        ffi.field = ff.Field(ffi.FF_FIELD)
    print "Polynom Evaluations"
    poly = [12, 6, 2, 1, 12, -17, -24, 5, 45, 1]
    poly_ff = a2ffa(poly)
    alphas_nff = [34, 12, 67, 3, 1, 89]
    alphas = a2ffa(alphas_nff)
    A = np.matrix(np.random.random_integers(0, field-1, (1200, 100)))
    An = np.split(A, 30)

    An_ff = [np.array(m2ffm(aa.tolist()), dtype=type(ffi.field(0))) for aa in An]

    st = []
    st_inv = []
    hs = []
    hs_inv = []
    so = []
    so_inv = []

    start = time.time()
    for al in alphas:
        st.append(standard_eval(An_ff, al, field))
        st_inv.append(standard_eval(An_ff, -al, field))
    stop = time.time()
    start_hs = time.time()
    for al in alphas:
        hs.append(horner_scheme(An_ff, al))
        hs_inv.append(horner_scheme(An_ff, -al))
    stop_hs = time.time()
    start_so = time.time()
    for al in alphas:
        one, another = second_order(An_ff, al, field)
        so.append(one)
        so_inv.append(another)
    stop_so = time.time()
    print [np.equal(st, hs) for i in range(len(st))]
    print [np.equal(st, so) for i in range(len(st))]
    print [np.equal(st_inv, hs_inv) for i in range(len(st_inv))]
    print [np.equal(st_inv, hs_inv) for i in range(len(st_inv))]

 #   print "Standard, values: ", st
 #   print "Horner, values: ", hs, hs_inv
 #   print "SO, values: ", so, so_inv

    print "Standard Evaluation: ", stop - start
    print "Horner Scheme: ", stop_hs - start_hs
    print "Second Order: ", stop_so - start_so
