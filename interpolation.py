from tools import *

counter = 0

def normalize(poly, field):
    zeros = None
    shape = None
    if poly:
        if isInstanceFF(poly[-1]):
            zeros = poly[-1] == setToZero()
        else:
            zeros = True
            for stroka in poly[-1]:
                for el in stroka:
                    if el != setToZero():
                        zeros = False
                        break
                if not zeros:
                    break
         #   zeros = poly[-1].all() == setToZero(field)
            shape = poly[-1].shape
    while poly and zeros:
        poly.pop()
    if poly == []:
        if shape:
            poly.append(np.array([[setToZero() for el in range(shape[1])] for stroka in range(shape[0])],
                                 dtype=type(setToZero())))
        else:
            poly.append(setToZero())


def poly_divmod(num, den, field):
    num = num[:]
 #   normalize(num, field)
    den = den[:]
   # normalize(den, field)


    if len(num) >= len(den):
        shiftlen = len(num) - len(den)
        den = [setToZero()] * shiftlen + den
    else:
        return [0], num

    quot = []
    divisor = den[-1]
    for i in range(shiftlen + 1):
        mult = divide(num[-1], divisor)
        quot = [mult] + quot

        zeros = None
        if isinstance(mult, int):
            non_zeros = mult != 0
        else:
            non_zeros = False
            for stroka in mult:
                for el in stroka:
                    if el != setToZero():
                        non_zeros = True
                        break
         #   non_zeros = mult.all() != 0
        if non_zeros:
            d = [multiply(mult, u) for u in den]
            num = [subtract(u, v) for u, v in zip(num, d)]

        num.pop()
        den.pop(0)

    #normalize(num, field)
    return quot, num


def divide(arr, el):
    return np.array([[arr_el / el for arr_el in row] for row in arr], dtype=type(el))


def subtract(arr1, arr2):
    return np.array([[el1 - el2 for el1, el2 in zip(u,v)] for u, v in zip(arr1, arr2)], dtype=type(arr1[0, 0]))


def multiply(arr, el):
    return np.array([[arr_el * el for arr_el in row] for row in arr], dtype=type(el))

def test(num, den):
    print ("%s / %s ->" % (num, den))
    q, r = poly_divmod(num, den)
    print ("quot: %s, rem: %s\n" % (q, r))
    return q, r


def createPolynom(roots):
    p = [1]
    for root in roots:
        p_str = [item*root for item in p]
        p = [0] + p
        p_out = []
        for u, v in zip(p_str, p[:-1]):
            p_out.append(v - u)
        p_out.append(p[-1])
        p = p_out
    return p


def a2ffa(arr):
    return ffi.array2ffarray(arr)


def m2ffm(C):
    return ffi.matrix2ffmatrix(C)


def cmp(a, b):
    return ffi.field.__cmp__(a, b)

def setToZero():
    return ffi.field(0)

def isInstanceFF(el):
    return isinstance(el, type(ffi.field(0)))


def modularForm(U, roots, field):
    global counter
    counter += 1
    print "Recursion: ", counter
    if not ffi.field:
        ffi.FF_FIELD = field
        ffi.field = ff.Field(ffi.FF_FIELD)

    values = []
  #  print "U", U
  #  print "roots", roots
    if len(roots) == 1:
        m11 = createPolynom(roots)
        first_arg = []
        for u in U:
            first_arg.append(np.asarray(m2ffm(u.tolist())))
        return poly_divmod(first_arg, a2ffa(m11), field)[1]
    if len(roots) <= len(U):
        n1 = len(roots) // 2
        roots_n1 = roots[:n1]
        roots_n2 = roots[n1:]

        devisor_1 = a2ffa(createPolynom(roots_n1))
        devisor_2 = a2ffa(createPolynom(roots_n2))
        devided = [np.array(m2ffm(u.tolist()), dtype=type(devisor_1[0])) for u in U]


        start_time = time.time()

        R1 = poly_divmod(devided, devisor_1, field)[1]
        pause_time = time.time()


        R2 = poly_divmod(devided, devisor_2, field)[1]
        stop_time = time.time()


        values += modularForm(R1, roots_n1, field)
        values += modularForm(R2, roots_n2, field)


        print"-------------------------------"
        print "R1: ", pause_time - start_time
        print "R2: ", stop_time - pause_time
        print"-------------------------------"

        return values
    else:
        n1 = len(roots) // 2
        roots_n1 = roots[:n1]
        roots_n2 = roots[n1:]
        values += modularForm(U, roots_n1, field)
        values += modularForm(U, roots_n2, field)
        return values



if __name__ == "__main__":
    print "hello world"
    A = np.matrix(np.random.random_integers(0, 65536, (1200, 100)))
    An = np.split(A, 30)
    alpha = [i+1 for i in range(30)]
    #alpha = [1,7]
    FIELD = 65537
    straight = []
    start = time.time()
    for al in alpha:
        straight.append(sum([An[index] * pow(al, index, FIELD) for index in range(len(An))]) % FIELD)
    stop = time.time()

    print "ans_straight ", straight
    print "An ", An
    start_pol = time.time()
    ans = modularForm(An, alpha, FIELD)
    stop_pol = time.time()

    print "ans ", ans
    print ([np.array_equal(ans[i], straight[i]) for i in range(len(ans))])

    print "Standard: ", stop - start
    print "Fast Modular Transforms via Division: ", stop_pol - start_pol


