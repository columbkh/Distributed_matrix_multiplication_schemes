def strassenR(A, B):
    n = len(A)

    newSize = n / 2
    a11 = [[0 for j in xrange(0, newSize)] for i in xrange(0, newSize)]
    a12 = [[0 for j in xrange(0, newSize)] for i in xrange(0, newSize)]
    a21 = [[0 for j in xrange(0, newSize)] for i in xrange(0, newSize)]
    a22 = [[0 for j in xrange(0, newSize)] for i in xrange(0, newSize)]

    b11 = [[0 for j in xrange(0, newSize)] for i in xrange(0, newSize)]
    b12 = [[0 for j in xrange(0, newSize)] for i in xrange(0, newSize)]
    b21 = [[0 for j in xrange(0, newSize)] for i in xrange(0, newSize)]
    b22 = [[0 for j in xrange(0, newSize)] for i in xrange(0, newSize)]

    aResult = [[0 for j in xrange(0, newSize)] for i in xrange(0, newSize)]
    bResult = [[0 for j in xrange(0, newSize)] for i in xrange(0, newSize)]

    top, down = np.split(A, 2)
    a11, a12 = np.split(top, 2, axis=1)
    a21, a22 = np.split(down, 2, axis=1)

    top, down = np.split(A, 2)
    b11, b12 = np.split(top, 2, axis=1)
    b21, b22 = np.split(down, 2, axis=1)

    # Calculating p1 to p7:
    p1 = (a11+a22) * (b11+b22) # p1 = (a11+a22) * (b11+b22)

    p2 = (a21+a22) * b11  # p2 = (a21+a22) * (b11)

    p3 = a11 * (b12 - b22)  # p3 = (a11) * (b12 - b22)

    p4 = a22 * (b21 - b11)  # p4 = (a22) * (b21 - b11)

    p5 = (a11+a12) * (b22)  # p5 = (a11+a12) * (b22)

    p6 = (a21-a11) * (b11+b12)  # p6 = (a21-a11) * (b11+b12)

    p7 = (a12-a22) * (b21+b22)  # p7 = (a12-a22) * (b21+b22)

    # calculating c21, c21, c11 e c22:
    c12 = p3 + p5  # c12 = p3 + p5
    c21 = p2 + p4  # c21 = p2 + p4
    c11 = p1 + p4 - p5 + p7  # c11 = p1 + p4 - p5 + p7
    c22 = p1 + p3 - p2 + p6  # c22 = p1 + p3 - p2 + p6

    top = np.concatenate((c11, c12), axis=1)
    down = np.concatenate((c21, c22), axis=1)
    C = np.concatenate((top, down))
    return C
