import numpy as np
import finite_field_comp as ff

FF_FIELD = 0

field = None

def getInvMatrix(rt, n, n_inv):
	x = [pow(rt, i, FF_FIELD) for i in range(n)]
	print x
	print len(x)
	print n
	for i in range(n):
		for j in range(1, n):
			print (n - ((j*i) % n)) % n
			
 	A_inv = [([1] + [x[(n - ((j*i) % n)) % n] for j in range(1, n)]) for i in range(n)]
 	print "+++++++"
 	print A_inv 
 	print "+++++++++"
	A = np.matrix(A_inv)
	print "A+++++++"
 	print A 
 	print "+++++++++"
	A = A.getT()
	A_inv = np.asarray(A)
	Aff = matrix2ffmatrix(A_inv)
	print "Aff+++++++"
 	print Aff 
 	print "+++++++++"
	n_invff = field(n_inv)
	print "Affxn+++++++"
 	print multMatrWithff(Aff, n_invff) 
 	print "+++++++++"
	return multMatrWithff(Aff, n_invff)
	
def getInvMatrix_with_x(x, n, n_inv):
	print x
	print len(x)
	print n
	for i in range(n):
		for j in range(1, n):
			print (n - ((j*i) % n)) % n
			
 	A_inv = [([1] + [x[(n - ((j*i) % n)) % n] for j in range(1, n)]) for i in range(n)]
 	print "+++++++"
 	print A_inv 
 	print "+++++++++"
	A = np.matrix(A_inv)
	print "A+++++++"
 	print A 
 	print "+++++++++"
	A = A.getT()
	A_inv = np.asarray(A)
	Aff = matrix2ffmatrix(A_inv)
	print "Aff+++++++"
 	print Aff 
 	print "+++++++++"
	n_invff = field(n_inv)
	print "Affxn+++++++"
 	print multMatrWithff(Aff, n_invff) 
 	print "+++++++++"
	return multMatrWithff(Aff, n_invff)
	
	
def multMatrWithff(matr, x):
	return [[col * x for col in row] for row in matr]


def matrix2ffmatrix(matr):
	return [[field(mat) for mat in mat_str] for mat_str in matr]
	
def array2ffarray(arr):
	return [field(el) for el in arr]

def ff_power(f_element, n):
	ret = field(1)
	for i in range(n):
		ret = ret * f_element
	return ret

def transposeMatrix(m):
    return map(list,zip(*m))

def getMatrixMinor(m,i,j):
    return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]

def getMatrixDeternminant(m):
    #base case for 2x2 matrix
    if len(m) == 2:
        return m[0][0]*m[1][1]-m[0][1]*m[1][0]

    determinant = field(0)
    for c in range(len(m)):
	#	print "determ is now ", str(determinant)
		determinant += ff_power(field(-1),c)*m[0][c]*getMatrixDeternminant(getMatrixMinor(m,0,c))
    return determinant
    

def getMatrixInverse(m):
    determinant = getMatrixDeternminant(m)
    print FF_FIELD
    #special case for 2x2 matrix:
    if len(m) == 2:
        return [[m[1][1]/determinant, field.__neg__(m[0][1]/determinant)],
                [field.__neg__(m[1][0]/determinant), m[0][0]/determinant]]

    #find matrix of cofactors
    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = getMatrixMinor(m,r,c)
            cofactorRow.append(ff_power(field(-1),(r+c)) * getMatrixDeternminant(minor))
        cofactors.append(cofactorRow)
    cofactors = transposeMatrix(cofactors)
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c]/determinant
    return cofactors
    
def swap_rows(m, i, j):
	tmp = m[i]
	m[i] = m[j]
	m[j] = tmp
	return m
	
def add_row(m, i, j, elem):
	tmp = []
	for el in m[j]:
		tmp.append(elem*el)
	for count in range(len(m[i])):
		m[i][count] = m[i][count] + tmp[count]
	return m
	
def mult_coeff(m, i, coeff):
	for count in range(len(m[i])):
		m[i][count] = m[i][count] * coeff
	return m 
	
def gaussian_elim(m):
	for i in range(len(m)):
		if m[i][i] == field(0):
			j = i+1
			while m[i][j] == 0:
				j += 1
				if j >= len(m):
					return -1
			m = swap_rows(m, i, j)
		m = mult_coeff(m, i, field.__invert__(m[i][i]))
		if i != 0:
			for count in range(i):
				m = add_row(m, count, i, field(-1) * m[count][i])
				if all(row == field(0) for row in m[count]): 
					return -1
		if i != len(m)-1:
			for count in range(i+1, len(m)):
				m = add_row(m, count, i, field(-1) * m[count][i])
				if all(row == field(0) for row in m[count]): 
					return -1
	return m
	
def inverse_matrix(m):
	ed = np.identity(len(m), dtype=int)
	ed_ff = matrix2ffmatrix(ed)
	app_m = [(m[count] + ed_ff[count]) for count in range(len(m))]
	gauss_m = gaussian_elim(app_m)
	if isinstance(gauss_m, int):
		if gauss_m == -1:
			return -1
	return [row[len(m):] for row in gauss_m]

if __name__ == "__main__":
	#A = np.random.rand(3, 3)
	#A = [[field(8), field(254)], [field(83), field(168)]]
	#A = [[field(167), field(253), field(187)], [field(2), field(4), field(166)], [field(142), field(57), field(44)]]
	k = 15
	rt = 3
	FF_FIELD = 32423
	field = ff.Field(FF_FIELD)
	x = [pow(rt, i, FF_FIELD) for i in range(k)]
	print x
	#C = np.random.random_integers(0, 100000, (5, 5))
    	
	C = [[pow(variable, i, FF_FIELD) for i in range(len(x))] for variable in x]
	A = matrix2ffmatrix(C)

	Am = np.matrix(A)
	Ainvm = np.matrix(np.asarray(inverse_matrix(A)))
	
	print Am
	print "-----------"

	print Ainvm
	
	print "-----------"
	B = (Am * Ainvm) % FF_FIELD
	
	print B

	
	#for i in range(len(A)):
	#	for j in range(len(A)):
	#		minor = getMatrixMinor(A, 1, 1)
			#Ainv = getMatrixInverse(A)
	#		print minor

