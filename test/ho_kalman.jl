include("../src/util.jl")
using LinearAlgebra: norm
using .Util: spectral_radius, ho_kalman

n = 50
m = 30
p = 20
T1 = 40
T2 = 60

A_raw = randn(Float64, (n, n))
#A_norm = LinearAlgebra.opnorm(A_raw)
#A = A_norm > 3 ? 0.9 * A_raw / A_norm : 0.5 * A_raw
#A = A_raw / LinearAlgebra.opnorm(A_raw)
A = 1.2 * A_raw / spectral_radius(A_raw)
B = rand(Float64, (n, p))
C = rand(Float64, (m, n))
D = rand(Float64, (m, p))
G = hcat(D, [C * A^k * B for k = 0 : T1+T2-1]...)

H = Util.make_hankel(G=G, p=p, T1=T1, T2=T2)
Ah, Bh, Ch = ho_kalman(H=H, n=n, m=m, p=p, T1=T1, T2=T2)

Gr = hcat(D, [Ch * Ah^k * Bh for k = 0 : T1+T2-1]...)
println(norm(G - Gr, Inf))
