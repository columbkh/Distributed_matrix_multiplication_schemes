import numpy as np
from tools import *

Zik = [[np.matrix(np.random.random_integers(0, 52, (2, 2))) for k in range(8)] for i in range(7)]

poly = Zik[0]
poly_ref = Zik[0][::-1]

field = 53

resp = second_order(poly_ref, 12, field)
st_enc = ((sum([(pow(12, k, field) * poly[k]) % field for k in range(0, 8)])) % field) % field
st_enc_neg = ((sum([(pow(field - 12, k, field) * poly[k]) % field for k in range(0, 8)])) % field) % field
