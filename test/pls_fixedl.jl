include("../src/util.jl")
include("../src/baselines.jl")
using .Util: synthetic_system, simulate, true_markov_params
using .Baselines: prefiltered_least_squares_fixedL
using LinearAlgebra: norm

n = 5
p = 3
m = 2
ρ = 1.0

σu = 1.0
σw = 0.1
σz = 0.1

T1 = 6
T2 = 4
T = T1 + T2 + 1
L = 10
μ = 1e-2
N = 100000

A, B, C, D = synthetic_system(n=n, p=p, m=m, ρ=ρ)
Y, U = simulate(A=A, B=B, C=C, D=D, σu=σu, σw=σw, σz=σz, T=N)
G = true_markov_params(A=A, B=B, C=C, D=D, T=T)
Gh = prefiltered_least_squares_fixedL(Y, U, T=T, L=L, μ=μ)
println(norm(G - Gh, Inf))
