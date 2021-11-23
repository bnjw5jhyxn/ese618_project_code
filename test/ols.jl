include("../src/util.jl")
include("../src/baselines.jl")
using .Util: synthetic_system, simulate, true_markov_params
using .Baselines: ordinary_least_squares
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
N = 100000

A, B, C, D = synthetic_system(n=n, p=p, m=m, ρ=ρ)
Y, U = simulate(A=A, B=B, C=C, D=D, σu=σu, σw=σw, σz=σz, T=N)
G = true_markov_params(A=A, B=B, C=C, D=D, T=T)
Gh = ordinary_least_squares(Y, U, T=T)
println(norm(G - Gh, Inf))
