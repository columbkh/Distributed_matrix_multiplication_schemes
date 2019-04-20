
k = [[1, 2, 5, 6, 8, 12, 18, 30, 36, 41], 
[1, 3, 13, 15, 25, 39],
[2, 4, 6, 14, 20, 26, 26],
[1, 2, 3, 6, 7, 11, 14, 17, 33, 42, 43]]

bfields = [3 * pow(2, kk) + 1 for kk in k[0]]
cfields = [5 * pow(2, kk) + 1 for kk in k[1]]
dfields = [3 * pow(2, kk) + 1 for kk in k[2]]
efields = [5 * pow(2, kk) + 1 for kk in k[3]]

res = [bfields, cfields, dfields, efields]
print res
