include("../src/util.jl")
include("../src/baselines.jl")
using .Util: synthetic_system, simulate, true_markov_params
using .Baselines: averaging, ordinary_least_squares, prefiltered_least_squares_fixedL
using LinearAlgebra: opnorm
using Statistics: mean
using Plots: plot, savefig, gr
using LaTeXStrings: @L_str

n = 5
p = 3
m = 2
#ρ = 0.9
#ρ = 0.9999
#ρ = 1.0001
ρ = 1.001

σu = 1.0
σwz = 0.25

T1 = 9
T2 = 8
T = T1 + T2 + 1
L = 9
μ = 1e-2
N0 = 200
trials_per_n = 20
@assert T * L + 1 ≤ 170

function averaging_error(N::Int64)::Vector{Float64}
    A, B, C, D = synthetic_system(n=n, p=p, m=m, ρ=ρ)
    Y, U = simulate(A=A, B=B, C=C, D=D, σu=σu, σw=σwz, σz=σwz, T=N)
    G = true_markov_params(A=A, B=B, C=C, D=D, T=T)
    [
     opnorm(averaging([(Y, U)], σu2=σu^2, T=T) - G),
     opnorm(ordinary_least_squares(Y, U, T=T) - G),
     opnorm(prefiltered_least_squares_fixedL(Y, U, T=T, L=L, μ=μ) - G),
    ]
end

Ns = N0 : 100 : 20000
errs = transpose(hcat([mean(hcat([averaging_error(N)
                                  for _ = 1:trials_per_n]...),
                           dims=2)
                       for N = Ns]...))
gr()
p = plot(
         Ns, errs,
         xlabel=L"N",
         ylabel=L"\Vert \hat G - G \Vert _{\rm op}",
         yscale=:log10,
         label=["avg" "OLS" "PLS"],
         title=L"error with $\rho(A) = %$ρ$ and $\sigma_w, \sigma_z = %$σwz$",
        )
savefig(p, "plots/baseline_error.svg")
