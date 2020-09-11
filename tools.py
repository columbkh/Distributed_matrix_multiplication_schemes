import math
import finite_field_inv as ffi
import finite_field_comp as ff
import numpy as np
import fft_fields
import communicators
import random
import sys
import time
import scipy.io as sio
from mpi4py import MPI

def make_x_for_a3s_so(N, field):
    x = []
    for i in range(int(math.floor(N/2))):
        xi = random.randint(1, field-1)
        x.append(xi)
        x.append(field-xi)
    if N % 2 != 0:
        x.append(random.randint(1, field-1))
    return x

def a2ffa(arr):
    return ffi.array2ffarray(arr)


def m2ffm(C):
    return ffi.matrix2ffmatrix(C)


def divide(arr, el):
    return np.array([[arr_el / el for arr_el in row] for row in arr], dtype=type(el))


def subtract(arr1, arr2):
    return np.array([[el1 - el2 for el1, el2 in zip(u,v)] for u, v in zip(arr1, arr2)], dtype=type(arr1[0, 0]))


def add(arr1, arr2):
    return np.array([[el1 + el2 for el1, el2 in zip(u,v)] for u, v in zip(arr1, arr2)], dtype=type(arr1[0, 0]))


def multiply_f(arr, el):
    return np.array([[arr_el * el for arr_el in row] for row in arr], dtype=type(el))



def horner_scheme(poly, x, field):
    shape = poly[-1].shape
    res = np.array([[0 for el in range(shape[1])] for stroka in range(shape[0])])
    for coef in poly:
        res = (res * x + coef) % field
    return res


def standard_eval(poly, x, field):
    shape = poly[-1].shape
    res = np.array([[0 for el in range(shape[1])] for stroka in range(shape[0])])
    for i in range(len(poly)):
        res = (res + poly[i] * pow(x, len(poly) - i - 1, field)) % field
    return res


def second_order(poly, x, field, f_x, g_x, field_matr):
    f_array = poly[::-1][0::2][::-1]
    g_array = poly[::-1][1::2][::-1]

    x_2 = pow(x, 2, field)
    i = 0

    for f in f_array:
        i = i + 1
        f_x = (f_x * x_2 + f)
        if i == 5:
            f_x = f_x % field
            i = 0

    f_x = f_x % field
    i = 0
    for g in g_array:
        i = i + 1
        g_x = (g_x * x_2 + g)
        if i == 4:
            g_x = g_x % field
            i = 0
    g_x = g_x * x % field

    x_pos = (f_x + g_x) % field
    x_neg = (f_x + field_matr - g_x) % field

    return x_pos, x_neg


def second_order_no_hs(poly, x, field, f_x, g_x, field_matr):
    f_array = poly[::-1][0::2]
    g_array = poly[::-1][1::2]

    x_2 = pow(x, 2, field)
    i = 0

    f_x = sum([f_array[i] * pow(x_2, i, field) for i in range(len(f_array))]) % field
    g_x = (sum([g_array[i] * pow(x_2, i, field) for i in range(len(g_array))]) * x) % field

    x_pos = (f_x + g_x) % field
    x_neg = (f_x + field_matr - g_x) % field

    return x_pos, x_neg


def cutoff_criterium(m, k, n):
    return 1.0 <= 22.0 * (4.0/n + 4.0/m + 7.0/k)


def strassen_winograd(A, B):
    n, k = A.shape
    k, m = B.shape

    if cutoff_criterium(m, k, n):
        return A * B

  #  if n <= LEAF_SIZE_N or k <= LEAF_SIZE_K or m<= LEAF_SIZE_M:
  #      return A * B
    if (m % 2 != 0) and (n % 2 != 0) and (k % 2 != 0):
        Atop, Adown = np.split(A, [n-1])
        A11, A12 = np.split(Atop, [k-1], axis=1)
        A21, A22 = np.split(Adown, [k-1], axis=1)

        Btop, Bdown = np.split(B, [k-1])
        B11, B12 = np.split(Btop, [m-1], axis=1)
        B21, B22 = np.split(Bdown, [m-1], axis=1)

        C11 = strassen_winograd(A11, B11)
        a12_b21 = np.dot(A12, B21)
        C11 += a12_b21
        C12 = np.dot(A11, B12) + np.dot(A12, B22)
        C21 = np.dot(A21, B11) + np.dot(A22, B21)
        C22 = np.dot(A21, B12) + np.dot(A22, B22)

        top_c = np.concatenate((C11, C12), axis=1)
        down_c = np.concatenate((C21, C22), axis=1)
        C = np.concatenate((top_c, down_c))
        return C

    if (n % 2 != 0) and (k % 2 != 0) and (m % 2 == 0):
        Atop, Adown = np.split(A, [n - 1])
        A11, A12 = np.split(Atop, [k - 1], axis=1)
        A21, A22 = np.split(Adown, [k - 1], axis=1)

        B11, B21 = np.split(B, [k - 1])

        C11 = strassen_winograd(A11, B11)
        a12_b21 = np.dot(A12, B21)
        C11 += a12_b21
        C21 = np.dot(A21, B11) + np.dot(A22, B21)

        C = np.concatenate((C11, C21))
        return C

    if (m % 2 != 0) and (k % 2 != 0) and (n % 2 == 0):
        A11, A12 = np.split(A, [k - 1], axis=1)
        Btop, Bdown = np.split(B, [k - 1])
        B11, B12 = np.split(Btop, [m - 1], axis=1)
        B21, B22 = np.split(Bdown, [m - 1], axis=1)

        C11 = strassen_winograd(A11, B11)
        a12_b21 = np.dot(A12, B21)
        C11 += a12_b21
        C12 = np.dot(A11, B12) + np.dot(A12, B22)

        C = np.concatenate((C11, C12), axis=1)
        return C

    if (m % 2 != 0) and (n % 2 != 0) and (k % 2 == 0):
        A11, A21 = np.split(A, [n-1])
        B11, B12 = np.split(B, [m - 1], axis=1)

        C11 = strassen_winograd(A11, B11)
        C12 = np.dot(A11, B12)
        C21 = np.dot(A21, B11)
        C22 = np.dot(A21, B12)

        top_c = np.concatenate((C11, C12), axis=1)
        down_c = np.concatenate((C21, C22), axis=1)
        C = np.concatenate((top_c, down_c))
        return C

    if (n % 2 != 0) and (m % 2 == 0) and (k % 2 == 0):
        A11, A21 = np.split(A, [n-1])

        C11 = strassen_winograd(A11, B)
        C21 = np.dot(A21, B)

        C = np.concatenate((C11, C21))
        return C

    if (m % 2 != 0) and (n % 2 == 0) and (k % 2 == 0):
        B11, B12 = np.split(B, [m - 1], axis=1)

        C11 = strassen_winograd(A, B11)
        C12 = np.dot(A, B12)

        C = np.concatenate((C11, C12), axis=1)
        return C

    if (k % 2 != 0) and (m % 2 == 0) and (n % 2 == 0):
        A11, A12 = np.split(A, [k - 1], axis=1)
        B11, B21 = np.split(B, [k - 1])

        C = strassen_winograd(A11, B11)
        a12_b21 = np.dot(A12, B21)
        C += a12_b21

        return C


    top, down = np.split(A, 2)
    a11, a12 = np.split(top, 2, axis=1)
    a21, a22 = np.split(down, 2, axis=1)

    top, down = np.split(B, 2)
    b11, b12 = np.split(top, 2, axis=1)
    b21, b22 = np.split(down, 2, axis=1)

    # Calculating p1 to p7:
    s1 = a21 + a22

    s2 = s1 - a11

    s3 = a11 - a21

    s4 = a12 - s2

    t1 = b12 - b11

    t2 = b22 - t1

    t3 = b22 - b12

    t4 = b21 - t2

    p1 = strassen_winograd(a11, b11) # p1 = (a11+a22) * (b11+b22)

    p2 = strassen_winograd(a12, b21)  # p2 = (a21+a22) * (b11)

    p3 = strassen_winograd(s1, t1)  # p3 = (a11) * (b12 - b22)

    p4 = strassen_winograd(s2, t2)  # p4 = (a22) * (b21 - b11)

    p5 = strassen_winograd(s3, t3)  # p5 = (a11+a12) * (b22)

    p6 = strassen_winograd(s4, b22)  # p6 = (a21-a11) * (b11+b12)

    p7 = strassen_winograd(a22, t4)  # p7 = (a12-a22) * (b21+b22)

    u1 = p1 + p2

    u2 = p1 + p4

    u3 = u2 + p5

    u4 = u3 + p7

    u5 = u3 + p3

    u6 = u2 + p3

    u7 = u6 + p6

    top = np.concatenate((u1, u7), axis=1)
    down = np.concatenate((u4, u5), axis=1)
    C = np.concatenate((top, down))
    return C


def strassen(A, B, LEAF_SIZE):
    n = len(A)

    if n <= LEAF_SIZE:
        return A * B
    top, down = np.split(A, 2)
    a11, a12 = np.split(top, 2, axis=1)
    a21, a22 = np.split(down, 2, axis=1)

    top, down = np.split(B, 2)
    b11, b12 = np.split(top, 2, axis=1)
    b21, b22 = np.split(down, 2, axis=1)

    # Calculating p1 to p7:
    p1 = strassen(a11+a22, b11+b22, LEAF_SIZE) # p1 = (a11+a22) * (b11+b22)

    p2 = strassen(a21+a22, b11, LEAF_SIZE)  # p2 = (a21+a22) * (b11)

    p3 = strassen(a11, b12 - b22, LEAF_SIZE)  # p3 = (a11) * (b12 - b22)

    p4 = strassen(a22, b21 - b11, LEAF_SIZE)  # p4 = (a22) * (b21 - b11)

    p5 = strassen(a11+a12, b22, LEAF_SIZE)  # p5 = (a11+a12) * (b22)

    p6 = strassen(a21-a11, b11+b12, LEAF_SIZE)  # p6 = (a21-a11) * (b11+b12)

    p7 = strassen(a12-a22, b21+b22, LEAF_SIZE)  # p7 = (a12-a22) * (b21+b22)

    # calculating c21, c21, c11 e c22:
    c12 = p3 + p5  # c12 = p3 + p5
    c21 = p2 + p4  # c21 = p2 + p4
    c11 = p1 + p4 - p5 + p7  # c11 = p1 + p4 - p5 + p7
    c22 = p1 + p3 - p2 + p6  # c22 = p1 + p3 - p2 + p6

    top = np.concatenate((c11, c12), axis=1)
    down = np.concatenate((c21, c22), axis=1)
    C = np.concatenate((top, down))
    return C


def check_array(lst, j, r, N):
    k = j % N
    for i in range(2 * r):
        if not (k + i in lst):
            return False
    return True

def set_communicatorsNlgasp(N, l, field):
    if not is_prime_number(field):
        print "Field is not prime"
        sys.exit(100)
    else:
        possb = get_for_fixedNl(N, l)
        if possb is False:
            print "No possabilities"
            sys.exit(100)
        else:
            r_a = possb.r_a
            r_b = possb.r_b

    if l >= min(r_a, r_b):
        inv_matr, an, ter, N_1, a, b = create_GASP_big(r_a, r_b, l, field)
    else:
        inv_matr, an, ter, N_1, a, b = create_GASP_small(r_a, r_b, l, field)


    communicators.prev_comm = MPI.COMM_WORLD
    if N_1 + 1 < communicators.prev_comm.Get_size():
        instances = [i for i in range(N_1 + 1, communicators.prev_comm.Get_size())]
        new_group = communicators.prev_comm.group.Excl(instances)
        communicators.gasp_comm = communicators.prev_comm.Create(new_group)
    else:
        communicators.gasp_comm = communicators.prev_comm


def set_communicatorsNl(N, l, field):
    communicators.prev_comm = MPI.COMM_WORLD
    if N + 1 < communicators.prev_comm.Get_size():
        instances = [i for i in range(N + 1, communicators.prev_comm.Get_size())]
        new_group = communicators.prev_comm.group.Excl(instances)
        communicators.comm = communicators.prev_comm.Create(new_group)
    else:
        communicators.comm = communicators.prev_comm


def set_comms_for_ass_fft(N):
    communicators.prev_comm = MPI.COMM_WORLD
    if N + 1 < communicators.prev_comm.Get_size():
        instances = [i for i in range(N + 1, communicators.prev_comm.Get_size())]
        new_group = communicators.prev_comm.group.Excl(instances)
        communicators.comm = communicators.prev_comm.Create(new_group)
    else:
        communicators.comm = communicators.prev_comm


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


def set_communicators_gasp_gaspr(r_a, r_b, l, r, field):
    if l >= min(r_a, r_b):
        inv_matr, an, ter, N, a, b = create_GASP_big(r_a, r_b, l, field)
    else:
        inv_matr, an, ter, N, a, b = create_GASP_small(r_a, r_b, l, field)

    inv_matr, an, ter, NN, a, b = create_GASP_r(r_a, r_b, l, r, field)

    communicators.prev_comm = MPI.COMM_WORLD
    if N + 1 < communicators.prev_comm.Get_size():
        instances = [i for i in range(N + 1, communicators.prev_comm.Get_size())]
        new_group = communicators.prev_comm.group.Excl(instances)
        communicators.comm = communicators.prev_comm.Create(new_group)
    else:
        communicators.comm = communicators.prev_comm

    if NN + 1 < communicators.prev_comm.Get_size():
        instances = [i for i in range(NN + 1, communicators.prev_comm.Get_size())]
        new_group_gr = communicators.prev_comm.group.Excl(instances)
        communicators.gr_comm = communicators.prev_comm.Create(new_group_gr)
    else:
        communicators.gr_comm = communicators.prev_comm


def interpol(missing, Crtn, field, kr, lst, var):
    for i in missing:
        print "missing", i
        coeff = [1] * kr
        for j in range(kr):
            for k in set(lst) - set([lst[j]]):
                coeff[j] = (coeff[j] * (var[i] - var[k]) * pow(var[lst[j]] - var[k], field - 2, field)) % field
        Crtn[i] = sum([Crtn[lst[j]] * coeff[j] for j in range(kr)]) % field


def lol_interpol(missing, Crtn, field, kr, lst, var):
    realCrtn = filter(lambda a: a != Crtn[lst[0]], Crtn)
    print "crtn", Crtn
    print "rc", realCrtn
    for i in missing:
        print "missing", i
        coeff = [1] * kr
        for j in range(kr):
            for k in set(lst) - set([lst[j]]):
                coeff[j] = (coeff[j] * (var[i] - var[k]) * pow(var[lst[j]] - var[k], field - 2, field)) % field
        Crtn[i] = sum([realCrtn[lst[j]] * coeff[j] for j in range(kr)]) % field


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
    an = make_a_n_for_so(N, r, field)
    delta = make_delta(field, an, r)
    delta_ff = arr2ffarray(delta, field)
    i_plus_an = [[i + ai for i in range(1, r + 1)] for ai in an]
    d_cross_left_part = make_d_cross_left_part(delta_ff, i_plus_an, N, field)
    d_cross_right_part = make_d_cross_right_part(delta_ff, an, N, field, l)
    d_cross = [d_cross_left_part[count] + d_cross_right_part[count] for count in range(N)]
    d_cross_inv = get_inv(d_cross, field)
    return d_cross_inv, np.asarray(d_cross_left_part), i_plus_an, an


def make_matrix_d_cross_so(N, field, r, l):
    an = make_a_n_for_so(N, r, field)
    delta = make_delta(field, an, r)
    delta_ff = arr2ffarray(delta, field)
    i_plus_an = [[i + ai for i in range(1, r + 1)] for ai in an]
    d_cross_left_part = make_d_cross_left_part(delta_ff, i_plus_an, N, field)
    d_cross_right_part = make_d_cross_right_part(delta_ff, an, N, field, l)
    d_cross = [d_cross_left_part[count] + d_cross_right_part[count] for count in range(N)]
    d_cross_inv = get_inv(d_cross, field)
    return d_cross_inv, np.asarray(d_cross_left_part), i_plus_an, an


def uscsa_make_matrix_d_cross_left_part_matrix(left_part, f, q):
    return [[matr[t // f][t % f] for t in range(f * q)] for matr in left_part]


def uscsa_make_matrix_d_cross_right_part(delta_ff, an, N, field, l, f):
    an_matr = [[pow(ai, i, field) for i in range(2 * l + f - 1)] for ai in an]
    an_ff = matr2ffmatrix(an_matr, field)
    return [[delta_ff[i] * el for el in an_ff[i]] for i in range(N)]


def uscsa_make_matrix_d_cross_left_part(delta_ff, pluss, N, field, q, f):
    pluss_ff = [matr2ffmatrix(matr, field) for matr in pluss]
    # print "this bitch ", pluss_ff
    # for n in range(N):
    #     for i in range(q):
    #         for j in range(f):
    #             print "j, i, n: ", j, i, n
    #             print "value of plussff: ", pluss_ff[n][i][j]
    #             print "delta value: ", delta_ff[n]
    #             print "result: ", delta_ff[n] / pluss_ff[n][i][j]
    return [[[delta_ff[n] / pluss_ff[n][i][j] for j in range(f)] for i in range(q)] for n in range(N)]


def uscsa_make_matrix_d_cross(N, field, q, f, l):
    an = make_a_n(N)
    #an = make_a_n_for_so(N, q, field)
    delta = make_delta(field, an, f*q)
    delta_ff = arr2ffarray(delta, field)
    j_plus_i_plus_an = [[[j + (i - 1) * f + ai for j in range(1, f + 1)] for i in range(1, q + 1)] for ai in an]
    i_plus_an = [[i + ai for i in range(1, q + 1)] for ai in an]
    d_cross_left_part = uscsa_make_matrix_d_cross_left_part(delta_ff, j_plus_i_plus_an, N, field, q, f)
    d_cross_left_part_matrix = uscsa_make_matrix_d_cross_left_part_matrix(d_cross_left_part, f, q)
    d_cross_right_part = uscsa_make_matrix_d_cross_right_part(delta_ff, an, N, field, l, f)
    d_cross = [d_cross_left_part_matrix[count] + d_cross_right_part[count] for count in range(N)]
    d_cross_inv = get_inv(d_cross, field)
    return d_cross_inv, np.asarray(d_cross_left_part), j_plus_i_plus_an, i_plus_an, an, delta


def uscsa_make_matrix_d_cross_so(N, field, q, f, l):
    #an = make_a_n(N)
    an = make_a_n_for_so(N, q, field)
    delta = make_delta(field, an, f*q)
    delta_ff = arr2ffarray(delta, field)
    j_plus_i_plus_an = [[[(j + (i - 1) * f + ai) % field for j in range(1, f + 1)] for i in range(1, q + 1)] for ai in an]
    i_plus_an = [[i + ai for i in range(1, q + 1)] for ai in an]
    d_cross_left_part = uscsa_make_matrix_d_cross_left_part(delta_ff, j_plus_i_plus_an, N, field, q, f)
    d_cross_left_part_matrix = uscsa_make_matrix_d_cross_left_part_matrix(d_cross_left_part, f, q)
    d_cross_right_part = uscsa_make_matrix_d_cross_right_part(delta_ff, an, N, field, l, f)
    d_cross = [d_cross_left_part_matrix[count] + d_cross_right_part[count] for count in range(N)]
    d_cross_inv = get_inv(d_cross, field)
    return d_cross_inv, np.asarray(d_cross_left_part), j_plus_i_plus_an, i_plus_an, an, delta




def so_encode_A(left_part, i_plus_an, A, field, N, l, r, field_matr, f_x, g_x, hs):
    Zik = [[np.matrix(np.random.random_integers(0, field - 1, (A.shape[0], A.shape[1]))) for k in range(l)] for i in
           range(r)]
    coded_matr = []
    all_coded_matr = []

    for n in range(N)[r::r+1]:
        resp = so_encode_An(left_part[n], i_plus_an[n], A, field, l, r, Zik, N, n, f_x, g_x, field_matr, hs)
        coded_matr.append(resp)

    if N % (r+1) != 0:
        coded_matr.append(so_encode_An(left_part[N-1], i_plus_an[N-1], A, field, l, r, Zik, N, N-1, f_x, g_x, field_matr, hs))

    T = N % (r+1)

    for n in range(N):
        if (n % (r+1) == r) or (n == N-1):
            all_coded_matr.append(coded_matr[n // (r+1)][0])
        elif (N - 1 - n) < T:
            t = N - 2 - n
            enc_matr = partly_encode_An(left_part[n], i_plus_an[n], A, field, l, r, Zik, t)
            enc_matr.insert(t, coded_matr[-1][1][t])
            all_coded_matr.append(enc_matr)
        else:
            t = n % (r + 1)
            enc_matr = partly_encode_An(left_part[n], i_plus_an[n], A, field, l, r, Zik, r-1-t)
            enc_matr.insert(r - (n % r) - 1, coded_matr[n // (r + 1)][1][r - (n % r) - 1])
            all_coded_matr.append(enc_matr)

    for n in range(N):
        for i in range(r):
            all_coded_matr[n][i] = (left_part[n][i] * all_coded_matr[n][i]) % field
    return all_coded_matr


def partly_encode_An(lpart, i_plus, A, field, l, r, Zik, n):
    return [(A + sum([(pow(i_plus[i], k, field) * Zik[i][k - 1]) % field for k in range(1, l + 1)])) % field for i in range(0, n)] \
           + [(A + sum([(pow(i_plus[i], k, field) * Zik[i][k - 1]) % field for k in range(1, l + 1)])) % field for i in range(n+1, r)]


def so_encode_An(lpart, i_plus, A, field, l, r, Zik, N, n, f_x, g_x, field_matr, hs):
    AA = []
    BB = []
    for i in range(r):
        poly = Zik[i][::-1] + [A]
        a, b = second_order(poly, i_plus[i], field, f_x, g_x, field_matr) if hs \
            else second_order_no_hs(poly, i_plus[i], field, f_x, g_x, field_matr)
        AA.append(a)
        BB.append(b)
    return AA, BB


def so_encode_B(i_plus_an, Bn, field, N, l, r, field_matr, f_x, g_x, hs):
    Zik = [[np.matrix(np.random.random_integers(0, field - 1, (Bn[0].shape[0], Bn[0].shape[1]))) for k in range(l)] for i in range(r)]
    coded_matr = []
    all_coded_matr = []

    for n in range(N)[r::r+1]:
        resp = so_encode_Bn(i_plus_an[n], Bn, field, r, Zik, f_x, g_x, field_matr, hs)
        coded_matr.append(resp)

    if N % (r+1) != 0:
        coded_matr.append(so_encode_Bn(i_plus_an[N-1], Bn, field, r, Zik, f_x, g_x, field_matr, hs))

    T = N % (r+1)

    for n in range(N):
        if (n % (r+1) == r) or (n == N-1):
            all_coded_matr.append(coded_matr[n // (r+1)][0])
        elif (N - 1 - n) < T:
            t = N - 2 - n
            enc_matr = partly_encode_Bn(i_plus_an[n], Bn, field, l, r, Zik, t)
            enc_matr.insert(t, coded_matr[-1][1][t])
            all_coded_matr.append(enc_matr)
        else:
            t = n % (r + 1)
            enc_matr = partly_encode_Bn(i_plus_an[n], Bn, field, l, r, Zik, r-1-t)
            enc_matr.insert(r - (n % r) - 1, coded_matr[n // (r + 1)][1][r - (n % r) - 1])
            all_coded_matr.append(enc_matr)

    return all_coded_matr


def partly_encode_Bn(i_plus, Bn, field, l, r, Zik, n):
    return [(Bn[i] + sum([(pow(i_plus[i], k, field) * Zik[i][k - 1]) % field for k in range(1, l + 1)])) % field for i in range(0, n)] \
           + [(Bn[i] + sum([(pow(i_plus[i], k, field) * Zik[i][k - 1]) % field for k in range(1, l + 1)])) % field for i in range(n+1, r)]


def so_encode_Bn(i_plus, Bn, field, r, Zik, f_x, g_x, field_matr, hs):
    AA = []
    BB = []
    for i in range(r):
        poly = Zik[i][::-1] + [Bn[i]]
        a, b = second_order(poly, i_plus[i], field, f_x, g_x, field_matr) if hs\
            else second_order_no_hs(poly, i_plus[i], field, f_x, g_x, field_matr)
        AA.append(a)
        BB.append(b)
    return AA, BB


def encode_An(lpart, i_plus, A, field, l, r, Zik):
    return [
        lpart[i] * ((A + sum([(pow(i_plus[i], k, field) * Zik[i][k - 1]) % field for k in range(1, l + 1)])) % field) % field
        for i in range(r)]


def encode_A(left_part, i_plus_an, A, field, N, l, r):
    start = time.time()
    Zik = [[np.matrix(np.random.random_integers(0, field-1, (A.shape[0], A.shape[1]))) for k in range(l)] for i in range(r)]
    stop = time.time()
    return [encode_An(left_part[n], i_plus_an[n], A, field, l, r, Zik) for n in range(N)]


def test_encode_A(left_part, i_plus_an, A, field, N, l, r):
    start = time.time()
    Zik = [[np.matrix(np.random.random_integers(0, field-1, (A.shape[0], A.shape[1]))) for k in range(l)] for i in range(r)]
    stop = time.time()
    return [encode_An(left_part[n], i_plus_an[n], A, field, l, r, Zik) for n in range(N)], stop - start


def reverse_encode_B(left_part, i_plus_an, B, field, N, l, r):
    Zik = [[np.matrix(np.random.random_integers(0, field-1, (B.shape[0], B.shape[1]))) for k in range(l)] for i in range(r)]
    return [reverse_encode_Bn(left_part[n], i_plus_an[n], B, field, l, r, Zik) for n in range(N)]


def reverse_encode_Bn(lpart, i_plus, B, field, l, r, Zik):
    return [
        lpart[i] * ((B + sum([(pow(i_plus[i], k, field) * Zik[i][k - 1]) % field for k in range(1, l + 1)])) % field)
        for i in range(r)]


def uscsa_encode_A(left_part, i_plus_an, Aj, field, N, l, f, q, delta):
    Zik = [[np.matrix(np.random.random_integers(0, field-1, (Aj[0].shape[0], Aj[0].shape[1]))) for k in range(l)] for i in range(q)]
    return [uscsa_encode_An(left_part[n], i_plus_an[n], Aj, field, l, q, f, Zik, delta[n]) for n in range(N)]


def uscsa_so_encode_B(left_part, i_plus_an, Bj, field, N, l, f, q, field_matr, f_x, g_x, hs):
    Zik = [[np.matrix(np.random.random_integers(0, field-1, (Bj[0].shape[0], Bj[0].shape[1]))) for k in range(l)] for i in
           range(q)]
    coded_matr = []
    all_coded_matr = []

    for n in range(N)[q::q+1]:
        resp = uscsa_so_encode_An(i_plus_an[n], field, q, Zik, f_x, g_x, field_matr, hs)
    #    print "enc so", resp
        coded_matr.append(resp)

    if N % (q+1) != 0:
        tmp = uscsa_so_encode_An(i_plus_an[N - 1], field, q, Zik, f_x, g_x, field_matr, hs)
   #     print "enc so", tmp
        coded_matr.append(tmp)

    T = N % (q+1)

    for n in range(N):
#        print "n ", n
        if (n % (q+1) == q) or (n == N-1):
            all_coded_matr.append(coded_matr[n // (q+1)][0])
        elif (N - 1 - n) < T:
            t = N - 2 - n
            enc_matr = uscsa_partly_encode_An(i_plus_an[n], field, l, q, Zik, t)
            enc_matr.insert(t, coded_matr[-1][1][t])
            all_coded_matr.append(enc_matr)
        else:
            t = n % (q + 1)
 #           print "partly enc ", q-1-t
  #          print "insert ", q - (n % q)
            enc_matr = uscsa_partly_encode_An(i_plus_an[n], field, l, q, Zik, q-1-t)
            enc_matr.insert(q-1-t, coded_matr[n // (q + 1)][1][q-1-t])
            all_coded_matr.append(enc_matr)


    for n in range(N):
        for i in range(q):
            all_coded_matr[n][i] = (Bj[i] + (multiply(left_part[n][i]) % field) * all_coded_matr[n][i]) % field

    return all_coded_matr


def reverse_gscsa_so_encode_A(left_part, i_plus_an, Bj, field, N, l, f, q, field_matr, f_x, g_x, hs):
    Zik = [[np.matrix(np.random.random_integers(0, field-1, (Bj[0].shape[0], Bj[0].shape[1]))) for k in range(l)] for i in
           range(q)]
    coded_matr = []
    all_coded_matr = []

    for n in range(N)[q::q+1]:
        resp = uscsa_so_encode_An(i_plus_an[n], field, q, Zik, f_x, g_x, field_matr, hs)
        coded_matr.append(resp)

    if N % (q+1) != 0:
        tmp = uscsa_so_encode_An(i_plus_an[N - 1], field, q, Zik, f_x, g_x, field_matr, hs)
        coded_matr.append(tmp)

    T = N % (q+1)

    for n in range(N):
#        print "n ", n
        if (n % (q+1) == q) or (n == N-1):
            all_coded_matr.append(coded_matr[n // (q+1)][0])
        elif (N - 1 - n) < T:
            t = N - 2 - n
            enc_matr = uscsa_partly_encode_An(i_plus_an[n], field, l, q, Zik, t)
            enc_matr.insert(t, coded_matr[-1][1][t])
            all_coded_matr.append(enc_matr)
        else:
            t = n % (q + 1)
            enc_matr = uscsa_partly_encode_An(i_plus_an[n], field, l, q, Zik, q-1-t)
            enc_matr.insert(q-1-t, coded_matr[n // (q + 1)][1][q-1-t])
            all_coded_matr.append(enc_matr)


    for n in range(N):
        for i in range(q):
            all_coded_matr[n][i] = (Bj[0] + (multiply(left_part[n][i]) % field) * all_coded_matr[n][i]) % field

    return all_coded_matr


def uscsa_so_encode_A(left_part, i_plus_an, Aj, field, N, l, f, q, delta, field_matr, f_x, g_x, hs):
    Zik = [[np.matrix(np.random.random_integers(0, field-1, (Aj[0].shape[0], Aj[0].shape[1]))) for k in range(l)] for i in
           range(q)]
    coded_matr = []
    all_coded_matr = []

    for n in range(N)[q::q+1]:
        resp = uscsa_so_encode_An(i_plus_an[n], field, q, Zik, f_x, g_x, field_matr, hs)
    #    print "enc so", resp
        coded_matr.append(resp)

    if N % (q+1) != 0:
        tmp = uscsa_so_encode_An(i_plus_an[N - 1], field, q, Zik, f_x, g_x, field_matr, hs)
   #     print "enc so", tmp
        coded_matr.append(tmp)

    T = N % (q+1)

    for n in range(N):
#        print "n ", n
        if (n % (q+1) == q) or (n == N-1):
            all_coded_matr.append(coded_matr[n // (q+1)][0])
        elif (N - 1 - n) < T:
            t = N - 2 - n
            enc_matr = uscsa_partly_encode_An(i_plus_an[n], field, l, q, Zik, t)
            enc_matr.insert(t, coded_matr[-1][1][t])
            all_coded_matr.append(enc_matr)
        else:
            t = n % (q + 1)
 #           print "partly enc ", q-1-t
  #          print "insert ", q - (n % q)
            enc_matr = uscsa_partly_encode_An(i_plus_an[n], field, l, q, Zik, q-1-t)
            enc_matr.insert(q-1-t, coded_matr[n // (q + 1)][1][q-1-t])
            all_coded_matr.append(enc_matr)


    for n in range(N):
        for i in range(q):
            all_coded_matr[n][i] = (sum([(left_part[n][i][j] * Aj[j]) % field for j in range(f)]) % field + delta[n] * all_coded_matr[n][i]) % field

    #print "second order "
    #for matr in all_coded_matr:
    #    print matr
    return all_coded_matr



def uscsa_partly_encode_An(i_plus, field, l, q, Zik, n):
    return [(sum([(pow(i_plus[i], k-1, field) * Zik[i][k-1]) % field for k in range(1, l + 1)])) % field for i in range(0, n)] \
           + [(sum([(pow(i_plus[i], k-1, field) * Zik[i][k-1]) % field for k in range(1, l + 1)])) % field for i in range(n+1, q)]


def reverse_uscsa_encode_B(left_part, i_plus_an, Bj, field, N, l, f, q, delta):
    Zik = [[np.matrix(np.random.random_integers(0, field-1, (Bj[0].shape[0], Bj[0].shape[1]))) for k in range(l)] for i in range(q)]
    return [reverse_uscsa_encode_Bn(left_part[n], i_plus_an[n], Bj, field, l, q, f, Zik, delta[n]) for n in range(N)]


def uscsa_so_encode_An(i_plus, field, q, Zik, f_x, g_x, field_matr, hs):
    AA = []
    BB = []
    for i in range(q):
        poly = Zik[i][::-1]
        a, b = second_order(poly, i_plus[i], field, f_x, g_x, field_matr) if hs\
            else second_order_no_hs(poly, i_plus[i], field, f_x, g_x, field_matr)
        AA.append(a)
        BB.append(b)
    return AA, BB


def uscsa_encode_An(lpart, i_plus, Aj, field, l, q, f, Zik, delta):
#    print "enc ", [sum([(pow(i_plus[i], k-1, field) * Zik[i][k-1]) % field for k in range(1, l + 1)]) % field for i in range(q)]
    return [(sum([(lpart[i][j] * Aj[j]) % field for j in range(f)]) % field + (delta * sum([(pow(i_plus[i], k-1, field) * Zik[i][k-1]) % field
                                                                                for k in range(1, l + 1)])) % field) % field for i in range(q)]


def reverse_gscsa_so_encode_B(left_part, i_plus_an, Bj, field, N, l, f, q, delta, field_matr, f_x, g_x, hs):
    Zik = [[np.matrix(np.random.random_integers(0, field-1, (Bj[0].shape[0], Bj[0].shape[1]))) for k in range(l)] for i in
           range(q)]
    coded_matr = []
    all_coded_matr = []

    for n in range(N)[q::q+1]:
        resp = uscsa_so_encode_An(i_plus_an[n], field, q, Zik, f_x, g_x, field_matr, hs)
    #    print "enc so", resp
        coded_matr.append(resp)

    if N % (q+1) != 0:
        tmp = uscsa_so_encode_An(i_plus_an[N - 1], field, q, Zik, f_x, g_x, field_matr, hs)
        coded_matr.append(tmp)

    T = N % (q+1)

    for n in range(N):
#        print "n ", n
        if (n % (q+1) == q) or (n == N-1):
            all_coded_matr.append(coded_matr[n // (q+1)][0])
        elif (N - 1 - n) < T:
            t = N - 2 - n
            enc_matr = uscsa_partly_encode_An(i_plus_an[n], field, l, q, Zik, t)
            enc_matr.insert(t, coded_matr[-1][1][t])
            all_coded_matr.append(enc_matr)
        else:
            t = n % (q + 1)

            enc_matr = uscsa_partly_encode_An(i_plus_an[n], field, l, q, Zik, q-1-t)
            enc_matr.insert(q-1-t, coded_matr[n // (q + 1)][1][q-1-t])
            all_coded_matr.append(enc_matr)


    for n in range(N):
        for i in range(q):
            all_coded_matr[n][i] = (sum([(left_part[n][i][j] * Bj[i*f + j]) % field for j in range(f)]) % field + delta[n] * all_coded_matr[n][i]) % field

    return all_coded_matr


def reverse_uscsa_encode_Bn(lpart, i_plus, Bj, field, l, q, f, Zik, delta):
    return [(sum([(lpart[i][j] * Bj[j]) % field for j in range(f)]) % field + (delta * sum([(pow(i_plus[i], k - 1, field) * Zik[i][k - 1]) % field
                                                                                for k in range(1, l + 1)])) % field) % field for i in range(q)]


def gscsa_encode_A(left_part, i_plus_an, Aj, field, N, l, f, q, delta):
    Zik = [[np.matrix(np.random.random_integers(0, field-1, (Aj[0].shape[0], Aj[0].shape[1]))) for k in range(l)] for i in range(q)]
    return [gscsa_encode_An(left_part[n], i_plus_an[n], Aj, field, l, q, f, Zik, delta[n]) for n in range(N)]


def gscsa_encode_An(lpart, i_plus, Aj, field, l, q, f, Zik, delta):
    return [(sum([(lpart[i][j] * Aj[i*f + j]) % field for j in range(f)]) % field + (delta * sum([(pow(i_plus[i], k-1, field) * Zik[i][k-1]) % field
                                                                                  for k in range(1, l + 1)])) % field) % field for i in range(q)]


def reverse_gscsa_encode_B(left_part, i_plus_an, Bj, field, N, l, f, q, delta):
    Zik = [[np.matrix(np.random.random_integers(0, field - 1, (Bj[0].shape[0], Bj[0].shape[1]))) for k in range(l)] for
           i in range(q)]
    return [reverse_gscsa_encode_Bn(left_part[n], i_plus_an[n], Bj, field, l, q, f, Zik, delta[n]) for n in range(N)]


def reverse_gscsa_encode_Bn(lpart, i_plus, Aj, field, l, q, f, Zik, delta):
    return [(sum([(lpart[i][j] * Aj[i*f + j]) % field for j in range(f)]) % field + (delta * sum([(pow(i_plus[i], k-1, field) * Zik[i][k-1]) % field
                                                                                  for k in range(1, l + 1)])) % field) % field for i in range(q)]


def encode_Bn(Bn, i_plus, field, l, r, Zik):
    return [(Bn[i] + sum([(pow(i_plus[i], k, field) * Zik[i][k - 1]) % field for k in range(1, l + 1)])) % field for i
            in range(r)]


def encode_B(Bn, i_plus_an, field, l, r, N):
    Zik = [[np.matrix(np.random.random_integers(0, field-1, (Bn[0].shape[0], Bn[0].shape[1]))) for k in range(l)] for i in
           range(r)]
    return [encode_Bn(Bn, i_plus_an[n], field, l, r, Zik) for n in range(N)]


def test_encode_B(Bn, i_plus_an, field, l, r, N):
    start = time.time()
    Zik = [[np.matrix(np.random.random_integers(0, field-1, (Bn[0].shape[0], Bn[0].shape[1]))) for k in range(l)] for i in
           range(r)]
    stop = time.time()
    return [encode_Bn(Bn, i_plus_an[n], field, l, r, Zik) for n in range(N)], stop - start


def reverse_encode_A(An, i_plus_an, field, l, r, N):
    Zik = [[np.matrix(np.random.random_integers(0, field-1, (An[0].shape[0], An[0].shape[1]))) for k in range(l)] for i in
           range(r)]
    return [reverse_encode_An(An, i_plus_an[n], field, l, r, Zik) for n in range(N)]


def reverse_encode_An(An, i_plus, field, l, r, Zik):
    return [(An[i] + sum([(pow(i_plus[i], k, field) * Zik[i][k - 1]) % field for k in range(1, l + 1)])) % field for i
            in range(r)]


def uscsa_encode_B(Bn, i_plus_an, field, l, q, f, N, left_term):
  #  delimet = [arr2ffarray([(l * pow(i_plus_an[4][i], k, field)) % field for k in range(l)], field) for i in range(q)]
  #  Zik_tmp = [[matr2ffmatrix(np.matrix(np.random.random_integers(1, 1, (Bn[0].shape[0], Bn[0].shape[1]))).tolist(), field) for k in range(l)] for i in
   #        range(q)]
    Zik = [[np.matrix(np.random.random_integers(0, field-1, (Bn[0].shape[0], Bn[0].shape[1]))) for k in range(l)] for i in
           range(q)]
 #   for i in range(q):
       # print "--------------"
       # print "i ", i
#        for k in range(l):
 #           for stroka in range(Bn[0].shape[0]):
  #              for stolbec in range(Bn[0].shape[1]):
   #                 Zik_tmp[i][k][stroka][stolbec] = Zik_tmp[i][k][stroka][stolbec] / delimet[i][k]
    #        Zik[i][k] = np.matrix(np.asarray(Zik_tmp[i][k]))
           # print "Zik ", Zik[i][k]

    return [uscsa_encode_Bn(Bn, i_plus_an[n], left_term[n], field, l, q, Zik) for n in range(N)]


def reverse_uscsa_encode_A(An, i_plus_an, field, l, q, f, N, left_term):
    Zik = [[np.matrix(np.random.random_integers(0, field-1, (An[0].shape[0], An[0].shape[1]))) for k in range(l)] for i in
           range(q)]
    return [reverse_uscsa_encode_An(An, i_plus_an[n], left_term[n], field, l, q, Zik) for n in range(N)]


def uscsa_encode_Bn(Bn, i_plus, lterm, field, l, q, Zik):
    return [(Bn[i] + (multiply(lterm[i]) % field) * sum([(pow(i_plus[i], k-1, field) * Zik[i][k-1]) % field for k in range(1, l + 1)])) % field for i in range(q)]


def reverse_uscsa_encode_An(An, i_plus, lterm, field, l, q, Zik):
    return [(An[i] + (multiply(lterm[i]) % field) * sum([(pow(i_plus[i], k-1, field) * Zik[i][k-1]) % field for k in range(1, l + 1)])) % field for i in range(q)]


def gscsa_encode_B(Bn, i_plus_an, field, l, q, f, N, left_term):
    Zik = [[np.matrix(np.random.random_integers(0, field-1, (Bn[0].shape[0], Bn[0].shape[1]))) for k in range(l)] for i in
           range(q)]
    return [gscsa_encode_Bn(Bn, i_plus_an[n], left_term[n], field, l, q, Zik) for n in range(N)]


def gscsa_encode_Bn(Bn, i_plus, lterm, field, l, q, Zik):
    return [(Bn[0] + (multiply(lterm[i]) % field) * sum([(pow(i_plus[i], k-1, field) * Zik[i][k-1]) % field for k in range(1, l + 1)])) % field for i in range(q)]


def reverse_gscsa_encode_A(An, i_plus_an, field, l, q, f, N, left_term):
    Zik = [[np.matrix(np.random.random_integers(0, field - 1, (An[0].shape[0], An[0].shape[1]))) for k in range(l)] for i in
           range(q)]
    return [reverse_gscsa_encode_An(An, i_plus_an[n], left_term[n], field, l, q, Zik) for n in range(N)]


def reverse_gscsa_encode_An(An, i_plus, lterm, field, l, q, Zik):
    return [(An[0] + (multiply(lterm[i]) % field) * sum([(pow(i_plus[i], k-1, field) * Zik[i][k-1]) % field for k in range(1, l + 1)])) % field for i in range(q)]


def getAencGASP(Ap, field, N, a, an):
    return [sum([(Ap[j] * pow(an[i], a[j], field)) % field for j in range(len(a))]) % field for i in range(N)]


def getBencGASP(Bp, field, N, b, an):
    return [sum([(Bp[j] * pow(an[i], b[j], field)) % field for j in range(len(b))]) % field for i in range(N)]


def make_a_n(N):
    return [i for i in range(N)]


def make_a_n_for_so(N, r, field):
    an = []
    tmp = 13
    for i in range(N):
        an.append(tmp)
        tmp = tmp + 2
 #   an[4] = field - an[4]
        if (i % (r+1) == r) or (i == N-1):
           an[i] = field - an[i]


  #  for i in range(N):
   #     if i % 2 == 0:
   #         an.append(i/2)
   #     else:
   #         an.append(-r-(i+1)/2)
    return an

def make_a_n_for_so_tmp(N, r, field, ind):
    an = []
    tmp = ind
    for i in range(N):
        an.append(tmp)
        tmp = tmp + 2

   #     if (i % (r+1) == r) or (i == N-1):
   #         an[i] = field - an[i]

    return an

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


def create_GASP_r(K, L, T, r, field):
    if K >= L:
        a = make_a_L_less_or_equal_K_r(K, L, T, r)
        b = make_b_L_less_or_equal_K_r(K, L, T)
    else:
        b = make_a_K_less_L_r(K, L, T)
        a = make_b_K_less_L_r(K, L, T, r)

    ter, N = terms(a, b)
    inv_matr, an = make_matrix(ter, N, field)
    return inv_matr, an, ter, N, a, b


def make_a_L_less_or_equal_K_r(K, L, T, r):
    first_part = [k - 1 for k in range(1, K + 1)]
    second_part = []
    count = 0

    for t in range(1, T+1):
        for rr in range(r):
            if count >= T:
                break
            second_part.append(K * L + K * (t - 1) + rr)
            count += 1

    return first_part + second_part


def make_a_K_less_L_r(K, L, T):
    return [K * (l - 1) for l in range(1, L + 1)] + [K * L + t - 1 for t in range(1, T + 1)]


def make_b_L_less_or_equal_K_r(K, L, T):
    return [K * (l - 1) for l in range(1, L + 1)] + [K * L + t - 1 for t in range(1, T + 1)]


def make_b_K_less_L_r(K, L, T, r):
    first_part = [k - 1 for k in range(1, K + 1)]
    second_part = []
    count = 0

    for t in range(1, T + 1):
        for rr in range(r):
            if count >= T:
                break
            second_part.append(K * L + K * (t - 1) + rr)
            count += 1

    return first_part + second_part


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
  #  print "d: ", d
    ordin = find_or(d, field)
    a = []
  #  print "ordin: ", ordin
    for i in range(1, ordin):
        s = lcm(i, ordin) / i
        if s == k:
         #   print "~i: ", i
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


def multiply(numbers):
    total = 1
    for x in numbers:
        total *= x
    return total


def get_all_possb_for_fixed_l(l):
    possbs = []
    for n in range(l+1, 100):
        if l > get_lmax(n):
            continue
        r_b = get_rb(n, l)
        r_a = get_ra(n, l, r_b)
        cand = (r_a + l) * (r_b + 1) - 1
        if is_power2(cand):
            possb = Params(n, l, r_a, r_b)
            possbs.append(possb)
    return possbs


def get_all_possb_for_fixedN(N):
    lmax = get_lmax(N)
    possbs = []
    print lmax
    for l in range(1, lmax):
        r_b = get_rb(N, l)
        r_a = get_ra(N, l, r_b)
        cand = (r_a + l) * (r_b + 1) - 1
        if is_power2(cand):
            possb = Params(N, l, r_a, r_b)
            possbs.append(possb)
    return possbs


def get_for_fixedNl(N, l):
    r_b = get_rb(N, l)
    r_a = get_ra(N, l, r_b)
    cand = (r_a + l) * (r_b + 1) - 1
    if is_power2(cand):
        return Params(N, l, r_a, r_b)
    else:
        return False


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


def getRestAenc(Ap, Ka, field, l, r_a, x):
    return [sum([Ap[j] * pow(xx, j, field) for j in range(r_a)]) % field + sum(
        [Ka[k] * pow(xx, k + r_a, field) for k in range(l)]) % field for xx in x]

def getRestReversedBenc(Bp, Kb, field, l, r_b, x):
    return [sum([Bp[j] * pow(xx, j, field) for j in range(r_b)]) % field + sum(
        [Kb[k] * pow(xx, k + r_b, field) for k in range(l)]) % field for xx in x]


def getAencSO(Ap, Ka, N, field, l, r_a, x, f_x, g_x, field_matr, hs):
    result = []
    coeff = Ka + Ap
    print "hs: ", hs
    for i in range(N-1)[::2]:
        a, b = second_order(coeff, x[i], field, f_x, g_x, field_matr) if hs\
            else second_order_no_hs(coeff, x[i], field, f_x, g_x, field_matr)
        result.append(a)
        result.append(b)
    if N % 2 == 1:
        res = sum([Ap[r_a - j - 1] * pow(x[N - 1], j, field) for j in range(r_a)]) % field + sum(
            [Ka[l - k - 1] * pow(x[N - 1], k + r_a, field) for k in range(l)]) % field

        result.append(
            res
        )
    return result


def getAenc(Ap, Ka, N, field, l, r_a, x):
    return [sum([Ap[j] * pow(x[i], j, field) for j in range(r_a)]) % field + sum(
        [Ka[k] * pow(x[i], k + r_a, field) for k in range(l)]) % field for i in range(N)]


def getBenc(Bp, Kb, N, field, l, r_a, r_b, x):
    return [sum([Bp[j] * pow(x[i], j * (r_a + l), field) for j in range(r_b)]) % field + sum(
        [Kb[k] * pow(x[i], k + r_a + (r_b - 1) * (r_a + l), field) for k in range(l)]) % field for i in range(N)]



def getReversedAenc(Ap, Ka, N, field, l, r_a, r_b, x):
    return [sum([Ap[j] * pow(x[i], j * (r_b + l), field) for j in range(r_a)]) % field + sum(
        [Ka[k] * pow(x[i], k + r_b + (r_a - 1) * (r_b + l), field) for k in range(l)]) % field for i in range(N)]


def getReversedBenc(Bp, Kb, N, field, l, r_b, x):
    return [sum([Bp[j] * pow(x[i], j, field) for j in range(r_b)]) % field + sum(
        [Kb[k] * pow(x[i], k + r_b, field) for k in range(l)]) % field for i in range(N)]


def getReversedBencSO(Bp, Kb, N, field, l, r_b, x, f_x, g_x, field_matr, hs):
    result = []
    coeff = Kb + Bp
    for i in range(N - 1)[::2]:
        a, b = second_order(coeff, x[i], field, f_x, g_x, field_matr) if hs\
            else second_order_no_hs(coeff, x[i], field, f_x, g_x, field_matr)
        result.append(a)
        result.append(b)
    if N % 2 == 1:
        res = sum([Bp[r_b - j - 1] * pow(x[N - 1], j, field) for j in range(r_b)]) % field + sum(
            [Kb[l - k - 1] * pow(x[N - 1], k + r_b, field) for k in range(l)]) % field

        result.append(
            res
        )
    return result



#def getNewAenc(Ap, Ka, N, field, l, r_a, r_b, x):
#    return [sum([Bp[j] * pow(x[i], j * (r_a + l), field) for j in range(r_b)]) % field + sum(
#        [Kb[k] * pow(x[i], k + r_a + (r_b - 1) * (r_a + l), field) for k in range(l)]) % field for i in range(N)]


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


def factorize_root(numb):
    sqrt_n = get_rounded(math.sqrt(numb))
    for q in reversed(range(1, sqrt_n + 1)):
        if numb % q == 0:
            return q, numb / q
    return numb, 1
