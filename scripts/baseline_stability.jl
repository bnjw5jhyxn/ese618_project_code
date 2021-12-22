include("../src/util.jl")
include("../src/baselines.jl")
include("../src/rank.jl")
include("../src/sysid.jl")
using .Util: synthetic_system, simulate, true_markov_params, make_hankel, ho_kalman
using .Baselines: averaging, ordinary_least_squares, prefiltered_least_squares_fixedL
using .Sysid: ordinary_dual_admm, prefiltered_dual_admm_fixedL
using LinearAlgebra: opnorm
using Statistics: mean
using Plots: plot, savefig, gr
using LaTeXStrings: @L_str

n = 5
p = 3
m = 2

σu = 1.0
σwz = 0.25

T1 = 9
T2 = 8
T = T1 + T2 + 1
L = 9
μ_pf = 1e-2
μ_nuc = 1e2
N = 10000
trials_per_ρ = 100
@assert T * L + 1 ≤ N

function experiment_error(ρ::Float64)::Vector{Float64}
    A, B, C, D = synthetic_system(n=n, p=p, m=m, ρ=ρ)
    Y, U = simulate(A=A, B=B, C=C, D=D, σu=σu, σw=σwz, σz=σwz, T=N)
    G = true_markov_params(A=A, B=B, C=C, D=D, T=T)
    [
     opnorm(averaging([(Y, U)], σu2=σu^2, T=T) - G),
     opnorm(ordinary_least_squares(Y, U, T=T) - G),
     opnorm(prefiltered_least_squares_fixedL(Y, U, T=T, L=L, μ=μ_pf) - G),
    ]
end

xs = LinRange(1.0, 4.0, 200)
ρs = 1 .- 10 .^ (-xs)
errs = transpose(hcat([mean(hcat([experiment_error(ρ)
                                  for _ = 1:trials_per_ρ]...),
                           dims=2)
                       for ρ = ρs]...))
gr()
p = plot(
         xs, log10.(errs),
         xlabel=L"\rho(A) = 1 - 10^{-x}",
         ylabel=L"\log_{10}\Vert \hat G - G \Vert _{\rm op}",
         label=["avg" "OLS" "PLS"],
         legend=:topleft,
         title=L"error with $N = %$N$ and $\sigma_w, \sigma_z = %$σwz$",
        )
savefig(p, "plots/stability.pdf")
