import math

N = 16384
# N = 37
# N = 101
rho = 0.5
n_conns = (N-1)*rho*N
w = 8
m, n = n_conns, n_conns
print(f'{m} pre and {n} post')
print('=============')


def calc_ref(m, n, rho, w): return m*rho*n*w
def calc_csr(m, n, rho, w):
    return m*math.log2(rho*m*n)+rho*m*n*(math.log2(n)+w)
def calc_bmp(m, n, rho, w): return m*math.log2(rho*m*n)+rho*m*n*w+m*n
def calc_cb(m, n, rho, w): return m*n*w


ref = calc_ref(m, n, rho, w)

csr = calc_csr(m, n, rho, w)

bmp = calc_bmp(m, n, rho, w)

cb = calc_cb(m, n, rho, w)

scores = {'cb': ref/cb,
          'csr': ref/csr,
          'bmp': ref/bmp}
mem_storage = {'cb': cb,
               'csr': csr,
               'bmp': bmp}

sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
sorted_stor = sorted(mem_storage.items(), key=lambda x: x[1])

print('efficiency')
print('=============')
for item, value in sorted_scores:
    print(item, value)

print('memory storage')
print('=============')
for item, value in sorted_stor:
    print(item, value)
