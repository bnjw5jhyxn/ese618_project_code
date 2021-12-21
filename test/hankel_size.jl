include("../src/util.jl")
using .Util: make_unclipped_hankel, hankel_adj, frob_inpr

m = 3
n = 4
j = 5
k = 6

X = randn(m, n * (j + k - 1))
Y = randn(m * j, n * k)
hankX = make_unclipped_hankel(X=X, n=n, j=j, k=k)
(mj, nk) = size(hankX)
@assert mj == m * j
@assert nk == n * k
adjY = hankel_adj(W=Y, m=m, n=n)
(m1, njkm1) = size(adjY)
@assert m1 == m
@assert njkm1 == n * (j + k - 1)
@assert frob_inpr(hankX, Y) ≈ frob_inpr(X, adjY)
