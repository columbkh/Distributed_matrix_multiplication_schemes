from tools import *


def a2ffa(arr):
    return ffi.array2ffarray(arr)


def horner_scheme(poly, x):
    res = ffi.field(0)
    x_ff = ffi.field(x)
    for coef in poly:
        res = res * x_ff + coef
    return res


def standard_eval(poly, x):
    res = ffi.field(0)
    for i in range(len(poly)):
        x_ff = ffi.field(pow(x, len(poly) - i - 1, field))
        res = res + poly[i] * x_ff
    return res


def second_order(poly, x):
    half_floor = math.floor((len(poly) - 1) / 2.0)
    half_ceil = math.ceil((len(poly) - 1) / 2.0)
    x_2 = ffi.field(pow(x, 2, field))
    x_ff = ffi.field(x)
    f_x = ffi.field(0)
    g_x = ffi.field(0)

    for i in range(int(2*half_ceil + 1))[1::2]:
        f_x = f_x * x_2 + poly[i]

    for i in range(int(2*half_floor + 1))[0::2]:
        g_x = g_x * x_2 + poly[i]
    g_x = g_x * x_ff


    x_pos = f_x + g_x
    x_neg = f_x - g_x
    return x_pos, x_neg

if __name__ == "__main__":
    field = 65537
    if not ffi.field:
        ffi.FF_FIELD = field
        ffi.field = ff.Field(ffi.FF_FIELD)
    print "Polynom Evaluations"
    poly = [12, 6, 2, 1, 12, -17, -24, 5, 45, 1]
    poly_ff = a2ffa(poly)

    while True:
        x = input("Put your x")
        start \
                = time.time()
        st = standard_eval(poly_ff, x)
        st_inv = standard_eval(poly_ff, -x)
        stop = time.time()
        start_hs = time.time()
        hs = horner_scheme(poly_ff, x)
        hs_inv = horner_scheme(poly_ff, -x)
        stop_hs = time.time()
        start_so = time.time()
        so, so_inv = second_order(poly_ff, x)
        stop_so = time.time()
        print np.equal(st, hs)
        print np.equal(st, so)
        print np.equal(st_inv, hs_inv)
        print np.equal(st_inv, so_inv)

        print "Standard, values: ", st, st_inv
        print "Horner, values: ", hs, hs_inv
        print "SO, values: ", so, so_inv

        print "Standard Evaluation: ", stop - start
        print "Horner Scheme: ", stop_hs - start_hs
        print "Second Order: ", stop_so - start_so
