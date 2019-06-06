import math
import finite_field_inv as ffi
import finite_field_comp as ff
import numpy as np
import fft_fields
import scipy.io as sio
import communicators
from mpi4py import MPI
import sys


def check_array(lst, j, r, N):
    k = j % N
    for i in range(2 * r):
        if not (k + i in lst):
            return False
    return True


def set_communicators(r_a, r_b, l, field):
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
            N = possb.N

    communicators.prev_comm = MPI.COMM_WORLD
    if N + 1 < communicators.prev_comm.Get_size():
        instances = [i for i in range(N + 1, communicators.prev_comm.Get_size())]
        new_group = communicators.prev_comm.group.Excl(instances)
        communicators.comm = communicators.prev_comm.Create(new_group)
    else:
        communicators.comm = communicators.prev_comm


def interpol(missing, Crtn, field, kr, lst, var):
    for i in missing:
        coeff = [1] * kr
        for j in range(kr):
            for k in set(lst) - set([lst[j]]):
                coeff[j] = (coeff[j] * (var[i] - var[k]) * pow(var[lst[j]] - var[k], field - 2, field)) % field
        Crtn[i] = sum([Crtn[lst[j]] * coeff[j] for j in range(kr)]) % field


def write_title_to_octavemnp(q, field, m, n, p, number, coeffinc):
    if type(coeffinc) == int:
        title = True
    else:
        title = False

    FrameStack = np.empty((8,), dtype=np.object)
    FrameStack[0] = title
    FrameStack[1] = q
    FrameStack[2] = field
    FrameStack[3] = m
    FrameStack[4] = n
    FrameStack[5] = p
    FrameStack[6] = number
    FrameStack[7] = coeffinc

    save_string = "./results/" + "title" + ".mat"
    sio.savemat(save_string, {"FrameStack": FrameStack})


def write_title_to_octave(q, field, matr_size, number, coeffinc):
    if type(coeffinc) == int:
        title = True
    else:
        title = False

    FrameStack = np.empty((6,), dtype=np.object)
    FrameStack[0] = title
    FrameStack[1] = q
    FrameStack[2] = field
    FrameStack[3] = matr_size
    FrameStack[4] = number
    FrameStack[5] = coeffinc

    save_string = "./results/" + "title" + ".mat"
    sio.savemat(save_string, {"FrameStack": FrameStack})


def write_to_octave(scheme, name):
    L = [li for li in scheme]
    FrameStack = np.empty((len(L),), dtype=np.object)
    for i in range(len(L)):
        FrameStack[i] = L[i]

    save_string = "./results/" + name + ".mat"
    sio.savemat(save_string, {"FrameStack": FrameStack})


def write_times(dec, dl, ul, serv_comp, file_name, iter_number):
    if iter_number == 0:
        f = open(file_name, 'w+')
    else:
        f = open(file_name, 'a+')

    f.write("Decoding %d: %f\r\n" % (iter_number, dec))
    f.write("---------------------------------------\r\n")
    if isinstance(ul, list):
        f.write("Uploading %d:\r\n" % iter_number)
        for i in range(len(ul)):
            f.write("Server Nr. %d: %f\r\n" % (i + 1, ul[i]))
        f.write("Average: %f\r\n" % (sum(ul) / len(ul)))
    else:
        f.write("Uploading %d: %f\r\n" % (iter_number, ul))

    if isinstance(dl, list):
        f.write("Downloading %d:\r\n" % iter_number)
        for i in range(len(dl)):
            f.write("Server Nr. %d: %f\r\n" % (i + 1, dl[i]))
        f.write("Average: %f\r\n" % (sum(dl) / len(dl)))
    else:
        f.write("Downloading %d: %f\r\n" % (iter_number, dl))

    if isinstance(serv_comp, list):
        f.write("Server Computation %d:\r\n" % iter_number)
        for i in range(len(serv_comp)):
            f.write("Server Nr. %d: %f\r\n" % (i + 1, serv_comp[i]))
        f.write("Average: %f\r\n" % (sum(serv_comp) / len(serv_comp)))
    else:
        f.write("Server Computation %d: %f\r\n" % (iter_number, serv_comp))

    f.write("---------------------------------------\r\n")
    f.close()


def print_times(dec, dl, ul, serv_comp):
    print "Decoding: ", dec
    if isinstance(ul, list):
        print "Uploading: "
        for i in range(len(ul)):
            print "Server Nr.", str(i + 1), ": ", ul[i]
        print "Average: ", str(sum(ul) / len(ul))
    else:
        print "Uploading: ", ul

    if isinstance(dl, list):
        print "Downloading: "
        for i in range(len(dl)):
            print "Server Nr.", str(i + 1), ": ", dl[i]
        print "Average: ", str(sum(dl) / len(dl))
    else:
        print "Downloading: ", dl

    if isinstance(serv_comp, list):
        print "Server Computation: "
        for i in range(len(serv_comp)):
            print "Server Nr.", str(i + 1), ": ", serv_comp[i]
        print "Average: ", str(sum(serv_comp) / len(serv_comp))
    else:
        print "Server Computation: ", serv_comp


def make_delta_i(field, ai, r):
    res = 1
    for u in range(1, r + 1):
        res *= (u + ai)
        res = res % field
    return res


def make_delta(field, an, r):
    return [make_delta_i(field, ai, r) for ai in an]


def make_d_cross_left_part(delta_ff, plus, N, field):
    plus_ff = matr2ffmatrix(plus, field)
    return [[delta_ff[i] / el for el in plus_ff[i]] for i in range(N)]


def make_d_cross_right_part(delta_ff, an, N, field, l):
    an_matr = [[pow(ai, i, field) for i in range(2 * l)] for ai in an]
    an_ff = matr2ffmatrix(an_matr, field)
    return [[delta_ff[i] * el for el in an_ff[i]] for i in range(N)]


def make_matrix_d_cross(N, field, r, l):
    an = make_a_n(N)
    delta = make_delta(field, an, r)
    delta_ff = arr2ffarray(delta, field)
    i_plus_an = [[i + ai for i in range(1, r + 1)] for ai in an]
    d_cross_left_part = make_d_cross_left_part(delta_ff, i_plus_an, N, field)
    d_cross_right_part = make_d_cross_right_part(delta_ff, an, N, field, l)
    d_cross = [d_cross_left_part[count] + d_cross_right_part[count] for count in range(N)]
    d_cross_inv = get_inv(d_cross, field)
    return d_cross_inv, np.asarray(d_cross_left_part), i_plus_an, an


def encode_An(lpart, i_plus, A, field, l, r, Zik):
    return [
        lpart[i] * ((A + sum([(pow(i_plus[i], k, field) * Zik[i][k - 1]) % field for k in range(1, l + 1)])) % field)
        for i in range(r)]


def encode_A(left_part, i_plus_an, A, field, N, l, r):
    Zik = [[np.matrix(np.random.random_integers(0, 0, (A.shape[0], A.shape[1]))) for k in range(l)] for i in range(r)]
    return [encode_An(left_part[n], i_plus_an[n], A, field, l, r, Zik) for n in range(N)]


def encode_Bn(Bn, i_plus, field, l, r, Zik):
    return [(Bn[i] + sum([(pow(i_plus[i], k, field) * Zik[i][k - 1]) % field for k in range(1, l + 1)])) % field for i
            in range(r)]


def encode_B(Bn, i_plus_an, field, l, r, N):
    Zik = [[np.matrix(np.random.random_integers(0, 0, (Bn[0].shape[0], Bn[0].shape[1]))) for k in range(l)] for i in
           range(r)]
    return [encode_Bn(Bn, i_plus_an[n], field, l, r, Zik) for n in range(N)]


def getAencGASP(Ap, field, N, a, an):
    return [sum([Ap[j] * pow(an[i], a[j], field) for j in range(len(a))]) for i in range(N)]


def getBencGASP(Bp, field, N, b, an):
    return [sum([Bp[j] * pow(an[i], b[j], field) for j in range(len(b))]) for i in range(N)]


def make_a_n(N):
    return [3 * i + 1 for i in range(N)]


def make_matrix(ter, N, field):
    flag = False
    while not flag:
        an = make_a_n(N)
        matr = [[pow(aa, t, field) for t in ter] for aa in an]
        res = get_inv(matr, field)
        flag = True
        if len(res) == 1:
            if res == -1:
                flag = False
    return res, an


def create_GASP_big(K, L, T, field):
    if K >= L:
        a = make_a_L_less_or_equal_K_big(K, L, T)
        b = make_b_L_less_or_equal_K_big(K, L, T)
    else:
        b = make_a_K_less_L_big(K, L, T)
        a = make_b_K_less_L_big(K, L, T)

    ter, N = terms(a, b)
    inv_matr, an = make_matrix(ter, N, field)
    return inv_matr, an, ter, N, a, b


def create_GASP_small(K, L, T, field):
    if K >= L:
        a = make_a_L_less_or_equal_K_small(K, L, T)
        b = make_b_L_less_or_equal_K_small(K, L, T)
    else:
        b = make_a_K_less_L_small(K, L, T)
        a = make_b_K_less_L_small(K, L, T)

    ter, N = terms(a, b)
    inv_matr, an = make_matrix(ter, N, field)
    return inv_matr, an, ter, N, a, b


def make_a_L_less_or_equal_K_big(K, L, T):
    return [k - 1 for k in range(1, K + 1)] + [K * L + t - 1 for t in range(1, T + 1)]


def make_a_K_less_L_big(K, L, T):
    return [K * (l - 1) for l in range(1, L + 1)] + [K * L + t - 1 for t in range(1, T + 1)]


def make_a_L_less_or_equal_K_small(K, L, T):
    return [k - 1 for k in range(1, K + 1)] + [K * (L + t - 1) for t in range(1, T + 1)]


def make_a_K_less_L_small(K, L, T):
    return [K * (l - 1) for l in range(1, L + 1)] + [K * L + t - 1 for t in range(1, T + 1)]


def make_b_L_less_or_equal_K_big(K, L, T):
    return [K * (l - 1) for l in range(1, L + 1)] + [K * L + t - 1 for t in range(1, T + 1)]


def make_b_K_less_L_big(K, L, T):
    return [k - 1 for k in range(1, K + 1)] + [K * L + t - 1 for t in range(1, T + 1)]


def make_b_L_less_or_equal_K_small(K, L, T):
    return [K * (l - 1) for l in range(1, L + 1)] + [K * L + t - 1 for t in range(1, T + 1)]


def make_b_K_less_L_small(K, L, T):
    return [k - 1 for k in range(1, K + 1)] + [K * (L + t - 1) for t in range(1, T + 1)]


def find_fft_field(field):
    if field in fft_fields.fields:
        return True
    else:
        return False


class Params:
    def __init__(self, N, l, r_a, r_b):
        self.r_a = r_a
        self.r_b = r_b
        self.N = N
        self.l = l
        self.k = (r_a + l) * (r_b + 1) - 1


def find_ord(field):
    return find_or(2, field)


def find_or(x, field):
    for i in range(1, field):
        res = pow(x, i, field)
        if res == 1:
            return i
    return -1


def gcd(x, y):
    while y:
        x, y = y, x % y
    return x


def is_prime_number(x):
    if x >= 2:
        for y in range(2, x):
            if not (x % y):
                return False
    else:
        return False
    return True


def find_pow(n):
    if not is_power2(n):
        return -1
    else:
        k = 0
        while n != 1:
            k += 1
            n = n >> 1
        return k


def find_invert_rt(n, field):
    ordin = find_ord(field)
    t = find_pow(n)
    return pow(2, ordin - t, field)


def lcm(x, y):
    lcm = (x * y) // gcd(x, y)
    return lcm


def arr2ffarray(arr, field):
    ffi.FF_FIELD = field
    ffi.field = ff.Field(ffi.FF_FIELD)
    return ffi.array2ffarray(arr)


def matr2ffmatrix(C, field):
    ffi.FF_FIELD = field
    ffi.field = ff.Field(ffi.FF_FIELD)
    return ffi.matrix2ffmatrix(C)


def find_rt(field, k):
    ordin = find_ord(field)
    a = []
    for i in range(1, ordin):
        s = lcm(i, ordin) / i
        if s == k:
            a.append(pow(2, i, field))
    return a


def haha_thats_funny(field, k):
    ordin = find_ord(field)
    u = lcm(k, ordin) / k
    return pow(2, u, field), u


def find_rt_any_d(field, k, d):
    ordin = find_or(d, field)
    a = []
    for i in range(1, ordin):
        s = lcm(i, ordin) / i
        if s == k:
            a.append(pow(d, i, field))
    return a


def get_dec_matr(x, field):
    C = [[pow(variable, i, field) for i in range(len(x))] for variable in x]
    return get_inv(C, field)


def find_inv_of_n(n, field):
    ordin = find_or(n, field)
    return ordin - 1


def terms_for_matrix(matr):
    values = []
    for zeile in matr:
        for row in zeile:
            if not (row in values):
                values.append(row)
    return values, len(values)


def terms(a, b):
    matr = [[aa + bb for bb in b] for aa in a]
    return terms_for_matrix(matr)


def find_sol(field, n):
    sol = [0]
    k = (field - 1) / gcd(field - 1, n)
    if k >= field - 1:
        return -1
    else:
        count = 2
        while len(sol) != n:
            while find_or(count, field) != field - 1 and pow(count, k, field) == sol[-1]:
                count += 1
            if count >= field:
                return -1
            sol.append((pow(count, k, field), count, find_or(count, field)))
            count += 1
        return sol


def decode_message(matr_inv, List, field):
    res = []
    for i in range(len(List)):
        res.append(sum([(matr_inv[i][j] * List[j]) % field for j in range(len(List))]) % field)
    return res


def get_inv_from_x(x, n, field):
    n_inv = find_inv_of_n(n, field)
    ffi.FF_FIELD = field
    ffi.field = ff.Field(ffi.FF_FIELD)
    return np.asarray(ffi.getInvMatrix_with_x(x, n, n_inv))


def get_inv_from_rt(rt, n, field):
    n_inv = find_inv_of_n(n, field)
    ffi.FF_FIELD = field
    ffi.field = ff.Field(ffi.FF_FIELD)
    return np.asarray(ffi.getInvMatrix(rt, n, n_inv))


def get_gauss_elim(C, field):
    ffi.FF_FIELD = field
    ffi.field = ff.Field(ffi.FF_FIELD)
    A = ffi.matrix2ffmatrix(C)
    res = ffi.gaussian_elim(A)
    return np.asarray(res)


def get_inv(C, field):
    ffi.FF_FIELD = field
    ffi.field = ff.Field(ffi.FF_FIELD)
    A = ffi.matrix2ffmatrix(C)
    res = ffi.inverse_matrix(A)
    if res == -1:
        return -1
    return np.asarray(res)


def get_rounded(number):
    res = math.ceil(number)
    return int(res)


def get_bottom(number):
    return get_rounded(number) - 1


def get_ceiling(number):
    return get_rounded(number)


def get_rb(N, l):
    sec_part = math.sqrt(1.0 / 4.0 + float(N) / float(l))
    res_float = -3.0 / 2.0 + sec_part
    res = get_rounded(res_float)
    return res


def get_lmax(N):
    return get_bottom(float(N - 1) / 2.0)


def get_ra(N, l, rb):
    cand_ra = 1
    eqw = (cand_ra + l) * (rb + 1) - 1
    while eqw <= N:
        res = cand_ra
        cand_ra += 1
        eqw = (cand_ra + l) * (rb + 1) - 1
    return res


def is_power2(number):
    return number != 0 and ((number & (number - 1)) == 0)


def find_var(field, k):
    for i in range(2, field):
        ordin = find_or(i, field)
        if ordin >= k:
            return i
    return -1


def get_all_possb_for_fixedN(N):
    lmax = get_lmax(N)
    possbs = []
    for l in range(1, lmax):
        r_b = get_rb(N, l)
        r_a = get_ra(N, l, r_b)
        cand = (r_a + l) * (r_b + 1) - 1
        if is_power2(cand):
            possb = Params(N, l, r_a, r_b)
            possbs.append(possb)
    return possbs


def get_nofft_for_fixedN(N, l):
    if l != 0:
        r_b = get_rb(N, l)
        r_a = get_ra(N, l, r_b)
        return Params(N, l, r_a, r_b)


def get_all_possabilities_nofft_for_fixedN(N):
    lmax = get_lmax(N)
    possbs = []
    for l in range(1, lmax):
        possbs.append(get_nofft_for_fixedN(N, l))
    return possbs


def get_all_possabilities_nofft(Nmax):
    all_possbs = []
    for worker_count in range(4, Nmax):
        possb = get_all_possabilities_nofft_for_fixedN(worker_count)
        all_possbs = all_possbs + possb
    return all_possbs


def get_all_possabilities(Nmax):
    all_possbs = []
    for worker_count in range(3, Nmax):
        possbs = get_all_possb_for_fixedN(worker_count)
        all_possbs = all_possbs + possbs
    return all_possbs


def getAenc(Ap, Ka, N, field, l, r_a, x):
    return [sum([Ap[j] * pow(x[i], j, field) for j in range(r_a)]) % field + sum(
        [Ka[k] * pow(x[i], k + r_a, field) for k in range(l)]) % field for i in range(N)]


def getBenc(Bp, Kb, N, field, l, r_a, r_b, x):
    return [sum([Bp[j] * pow(x[i], j * (r_a + l), field) for j in range(r_b)]) % field + sum(
        [Kb[k] * pow(x[i], k + r_a + (r_b - 1) * (r_a + l), field) for k in range(l)]) % field for i in range(N)]


def inverse_n(n, rt, field):
    if rt == n:
        return pow(rt, n - 1, field)
    else:
        if rt > n:
            s = rt / n
            return pow(n, s * n - 1, field)
        else:
            s = n / rt
            return pow(n, s * n - 1, field)


def fast_fourier_transform(n, A, rt, field):
    if n == 1:
        return A
    else:
        B = fast_fourier_transform(n / 2, A[0:][::2], pow(rt, 2, field), field)
        C = fast_fourier_transform(n / 2, A[1:][::2], pow(rt, 2, field), field)
        Afft = []

        for i in range(n / 2):
            tmp = (B[i] + pow(rt, i, field) * C[i]) % field
            Afft.insert(i, tmp)
            tmp = (B[i] + pow(rt, i + n / 2, field) * C[i]) % field
            Afft.insert(i + n / 2, tmp)

        return Afft


def inverse_fft(n, A, rt, field):
    rt_inv = pow(rt, n - 1, field)
    Afft = fast_fourier_transform(n, A, rt_inv, field)
    return [(Afft[i] * find_invert_rt(n, field)) % field for i in range(len(Afft))]
