from tools import *
import argparse
import random
from mpi4py import MPI
import time
import sys

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--Field', type=int, help='Finite Field')
parser.add_argument('--N', type=int, help='number of workers')
parser.add_argument('--l', type=int, help='l')
parser.add_argument('--barrier', help='Enable barrier', action="store_true")
parser.add_argument('--verific', help='Enable Verification', action="store_true")
parser.add_argument('--all_together', help='Compute all together', action = "store_true")

args = parser.parse_args()

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
	
	
N = args.N
l = args.l
r = N - 2 * l
Field = args.Field

m = 500
n = 500
p = 500

if p % r != 0:
	p = (p // r) * r
	
prev_comm = MPI.COMM_WORLD
if N+1 < prev_comm.Get_size():
     print "More"
     instances = [i for i in range(N+1, prev_comm.Get_size())]
     new_group = prev_comm.group.Excl(instances)
     comm = prev_comm.Create(new_group)
else:
     print "Less"
     comm = prev_comm

	


if prev_comm.rank == 0:  # Master
    print "Running with %d processes:" % comm.Get_size()
	
    print "r: ", r
    print "N: ", N
    print "l: ", l
    
    if N > 19:
		print "Too many instances"
		sys.exit(100)
        
    A = np.matrix(np.random.random_integers(0, 255, (m, n)))
    B = np.matrix(np.random.random_integers(0, 255, (p, n)))
	
    Bn = np.split(B, r)
	
	
    dec_start = time.time()
	
    d_cross, left_part, i_plus_an, an = make_matrix_d_cross(N, Field, r, l)
    
    Aenc = encode_A(left_part, i_plus_an, A, Field, N, l, r)
    Benc = encode_B(Bn, i_plus_an, Field, l, r, N)

    dec_pause = time.time()
    dec_firstpart = dec_pause - dec_start
	
    Crtn = []
    serv_comp = [None] * N
    ul_stop = [None] * N
    #dl = [None] * N
    for i in range(N):
		Crtn.append(np.zeros((m, p/r), dtype=np.int_))
	
    reqA = [None] * N * r
    reqB = [None] * N * r
    reqC = [None] * N
    reqS = [None] * N
    
    if together:
	ul_start = time.time()
	for i in range(N):
			for j in range(r):
				reqA[i+j*N] = comm.Isend([Aenc[i][j], MPI.INT], dest=i+1, tag=15)
				reqB[i+j*N] = comm.Isend([Benc[i][j], MPI.INT], dest=i+1, tag=29)
			reqC[i] = comm.Irecv([Crtn[i], MPI.INT], source=i+1, tag=42)
		
	MPI.Request.Waitall(reqA)
        MPI.Request.Waitall(reqB)
        
        #ul_stop = time.time()
		#ul = ul_stop - ul_start
		
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
		reqAB = [None] * 2 * N * r
		
		for i in range(N):
			ul_start[i] = time.time()
			for j in range(r):
				reqAB[i+j*N] = comm.Isend([Aenc[i][j], MPI.INT], dest=i+1, tag=15)
				reqAB[i+(j+r)*N] = comm.Isend([Benc[i][j], MPI.INT], dest=i+1, tag=29)
			reqC[i] = comm.Irecv([Crtn[i], MPI.INT], source=i+1, tag=42)

	#	lst = []
		
	#	for i in range(N*2*r):
#			j = MPI.Request.Waitany(reqAB)
#			lst.append(j)
#			if check_array(lst, j, r, N):
#				ul_stop[j % N] = time.time()
 ##   
   #     for i in range(N):
	#		ul[i] = ul_stop[i] - ul_start[i]
			
		if barrier:
			comm.Barrier()
			
		dl_start = time.time()
			
		for i in range(N):
			j = MPI.Request.Waitany(reqC)
			tmp = time.time()
			dl[j] = tmp - dl_start
		
		
    
    dec_pause = time.time()
    
    res = decode_message(d_cross, Crtn, Field)
    
    final_res = res[0:r]
    
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
		Cver = [(A * bb.getT()) % Field for bb in Bn[:r]]
		print ([np.array_equal(final_res[i], Cver[i]) for i in range(len(Cver))])


if prev_comm.rank > 0 and prev_comm.rank < N+1:
    Ai = []
    Bi = []
    rA = [None] * r
    rB = [None] * r
    Ci = np.empty_like(np.matrix([[0]*(p/r) for i in range(m)]))
    for j in range(r):
		Aij = np.empty_like(np.matrix([[0]*n for i in range(m)]))
                Bij = np.empty_like(np.matrix([[0]*n for i in range(p/r)]))
		#rA = comm.Irecv(Aij, source=0, tag=15)
		#rB = comm.Irecv(Bij, source=0, tag=29)
		rA[j] = comm.Irecv(Aij, source=0, tag=15)
		rB[j] = comm.Irecv(Bij, source=0, tag=29)
		Ai.append(Aij)
		Bi.append(Bij)
		
	#rA.wait()
    #rB.wait()
    MPI.Request.Waitall(rA)
    MPI.Request.Waitall(rB)
    
    print("Worker " + str(comm.rank) + " received the message from master")

    
    servcomp_start = time.time()
     
    for j in range(r):
		Ci += (Ai[j] * (Bi[j].getT()))
		Ci = Ci % Field
	
    servcomp_done = time.time()
    
    servcomp = servcomp_done - servcomp_start
		
    if barrier:
		comm.Barrier()
		
    sC = comm.Isend(Ci, dest=0, tag=42)
    sC.Wait()
    
    #downl_sent = time.time()
    
    print("Worker " + str(comm.rank) + " sent the result to master")

    
    if barrier:
		comm.Barrier()
		
    sSrv = comm.send(servcomp, dest=0, tag=64) 
    sUpl = comm.send(servcomp_start, dest=0, tag=70)
	#sDwl = comm.send(downl_sent, dest=0, tag=80)

    


















