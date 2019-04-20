
from tools import *
import argparse
import random
from mpi4py import MPI
import time
import sys

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--r_a', type=int, help='divide A on K')
parser.add_argument('--r_b', type=int, help='divide B on L')
parser.add_argument('--l', type=int, help='number of colluding workers')
parser.add_argument('--Field', type=int, help="Field")
parser.add_argument('--barrier', help='Enable barrier', action="store_true")
parser.add_argument('--verific', help='Enable Verification', action="store_true")
parser.add_argument('--all_together', help='Compute all together', action = "store_true")
args = parser.parse_args()


r_a = args.r_a
r_b = args.r_b
l = args.l
Field = args.Field
if args.barrier:
	barrier = True
else:
	barrier = False
	
if args.verific:
	verific = True
else:
	verific = False

if args.all_together:
	together = True
else:
	together = False
	
if l > min(r_a, r_b):
	print "Bad arguments"
	sys.exit(100)
	

m = 1000
n = 1000
p = 1000

if m % r_a != 0:
	m = (m // r_a) * r_a
if p % r_b != 0:
	p = (p // r_b) * r_b
	
comm = MPI.COMM_WORLD

if comm.rank == 0:  # Master
	print "Running with %d processes:" % comm.Get_size()
	
	print "m: ", m
	print "p: ", p
	print "n: ", n
    print "r_a: ", r_a
    print "r_b: ", r_b
   # print "N: ", N
    print "l: ", l
    
    A = np.matrix(np.random.random_integers(0, 255, (m, n)))
	B = np.matrix(np.random.random_integers(0, 255, (p, n)))
	
	Ap = np.split(A, r_a)
	Bp = np.split(B, r_b)
	
	Ka = [np.matrix(np.random.random_integers(0, 255, (m/r_a, n))) for i in range(l)]
	Kb = [np.matrix(np.random.random_integers(0, 255, (p/r_b, n))) for i in range(l)]
	
	Ap += Ka
	Bp += Kb
    
    dec_start = time.time()
    
    if l == min(r_a, r_b):
		inv_matr, an, ter, N, a, b = create_GASP_big(r_a, r_b, l, Field)
    else:
		inv_matr, an, ter, N, a, b = create_GASP_small(r_a, r_b, l, Field)
		
	print "N: ", N
    
	
	Aenc = getAencGASP(Ap, Field, N, a, an)
	Benc = getBencGASP(Bp, Field, N, b, an)
	
    dec_pause = time.time()
    dec_firstpart = dec_pause - dec_start
    
    if N > 19:
	     print "Too many instances"
	     sys.exit(100)
    
    Crtn = []
    serv_comp = [None] * N
    ul_stop = [None] * N
	for i in range(N):
		Crtn.append(np.zeros((m/r_a, p/r_b), dtype=np.int_))
	
    reqA = [None] * N
    reqB = [None] * N
    reqC = [None] * N
    reqS = [None] * N
    
    
    if together:
		ul_start = time.time()
		for i in range(N):
			reqA[i] = comm.Isend([Aenc[i], MPI.INT], dest=i+1, tag=15)
			reqB[i] = comm.Isend([Benc[i], MPI.INT], dest=i+1, tag=29)
			reqC[i] = comm.Irecv([Crtn[i], MPI.INT], source=i+1, tag=42)
			
		MPI.Request.Waitall(reqA)
		MPI.Request.Waitall(reqB)
		
#		ul_stop = time.time()
#		ul = ul_stop - ul_start
		
		if barrier:
			comm.Barrier()
	
	    dl_start = time.time()
	    MPI.Request.Waitall(reqC)
	    
	    dl_stop = time.time()
	    dl = dl_stop - dl_start
	    
	else:
		ul_start = [None] * N
	#	ul_stop = [None] * N
	#	ul = [None] * N
		dl = [None] * N
		reqAB = [None] * 2 * N
		for i in range(N):
			ul_start[i] = time.time()
			reqAB[i] = comm.Isend([Aenc[i], MPI.INT], dest=i+1, tag=15)
			reqAB[i + N] = comm.Isend([Benc[i], MPI.INT], dest=i+1, tag=29)
			reqC[i] = comm.Irecv([Crtn[i], MPI.INT], source=i+1, tag=42)
			
#		lst = []
#		
#		for i in range(N*2):
#			j = MPI.Request.Waitany(reqAB)
#			lst.append(j)
#			if j < N:
#				if j + N in lst:
#					ul_stop[j] = time.time()
#			else:
#				if j - N in lst:
#					ul_stop[j-N] = time.time()
    
#        for i in range(N):
#			ul[i] = ul_stop[i] - ul_start[i]
			
		if barrier:
			comm.Barrier()
			
		dl_start = time.time()
		
		for i in range(N):
			j = MPI.Request.Waitany(reqC)
			tmp = time.time()
			dl[j] = tmp - dl_start
			
			
		
		
		
			
			    
    dec_pause = time.time()
    
    res = decode_message(inv_matr, Crtn, Field)
    
    final_res = res[:r_b]
    
    for k in range(r_a-1):
		final_res += res[k * (1 + r_b) + r_b + l:(k + 1) * (r_b + 1) + l + r_b - 1]
    
    dec_done = time.time()
    
    dec_secondpart = dec_done - dec_pause
    dec = dec_firstpart + dec_secondpart
    
    
    if barrier:
		comm.Barrier()
	
	for i in range(N):	
		serv_comp[i] = comm.recv(source=i+1, tag=64)
    for i in range(N):	
		ul_stop[i] = comm.recv(source=i+1, tag=70)
		
	if together:
		ul_stop_latest = max(ul_stop)
		ul = ul_stop_latest - ul_start
	else:
		ul = [None] * N
		for i in range(N):
		    ul[i] = ul_stop[i] - ul_start[i]
    
    print_times(dec, dl, ul, serv_comp)

    if verific:
		Cver = []
		for aa in Ap[:r_a]:
			Cver += [(aa * bb.getT()) % Field for bb in Bp[:r_b]]
    
        print ([np.array_equal(final_res[i], Cver[i]) for i in range(len(Cver))])
    
    
    
    
    
    
else:  # Worker
	Ai = np.empty_like(np.matrix([[0]*n for i in range(m/r_a)]))
    Bi = np.empty_like(np.matrix([[0]*n for i in range(p/r_b)]))
    rA = comm.Irecv(Ai, source=0, tag=15)
    rB = comm.Irecv(Bi, source=0, tag=29)

    rA.wait()
    rB.wait()
    
    print("Worker " + str(comm.rank) + " received the message from master")
    
    servcomp_start = time.time()
    
    Ci = (Ai * (Bi.getT())) % Field
    
    servcomp_done = time.time()
    
    servcomp = servcomp_done - servcomp_start
    
    if barrier:
		comm.Barrier()
    
    sC = comm.Isend(Ci, dest=0, tag=42)
    sC.Wait()
    print("Worker " + str(comm.rank) + " sent the result to master")
    
    if barrier:
		comm.Barrier()
		
	sSrv = comm.send(servcomp, dest=0, tag=64)
   	sUpl = comm.send(servcomp_start, dest=0, tag=70)


















