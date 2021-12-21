include("../src/util.jl")
include("../src/baselines.jl")
include("../src/rank.jl")
include("../src/sysid.jl")
using .Util: synthetic_system, simulate, true_markov_params, make_hankel, ho_kalman
using .Baselines: averaging, ordinary_least_squares, prefiltered_least_squares_fixedL
using .Sysid: ordinary_dual_admm
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
μ_nuc = 1e3
N = 20000
trials_per_ρ = 10
@assert T * L + 1 ≤ N

function ho_kalman_error(;
        G::Matrix{Float64},
        Gh::Matrix{Float64},
    )::Float64
    Ah, Bh, Ch = ho_kalman(H=make_hankel(G=G, p=p, T1=T1, T2=T2),
                           n=n, m=m, p=p, T1=T1, T2=T2)
    Dh = Gh[:, 1:p]
    opnorm(true_markov_params(A=Ah, B=Bh, C=Ch, D=Dh, T=T) - G)
end

function experiment_error(ρ::Float64)::Vector{Float64}
    A, B, C, D = synthetic_system(n=n, p=p, m=m, ρ=ρ)
    Y, U = simulate(A=A, B=B, C=C, D=D, σu=σu, σw=σwz, σz=σwz, T=N)
    G = true_markov_params(A=A, B=B, C=C, D=D, T=T)
    [
     ho_kalman_error(G=G, Gh=averaging([(Y, U)], σu2=σu^2, T=T)),
     ho_kalman_error(G=G, Gh=ordinary_least_squares(Y, U, T=T)),
     ho_kalman_error(G=G, Gh=prefiltered_least_squares_fixedL(Y, U, T=T, L=L, μ=μ_pf)),
     ho_kalman_error(G=G, Gh=ordinary_dual_admm(Y, U, T1=T1, T2=T2, μ=μ_nuc)),
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
         label=["avg" "OLS" "PLS" "admm"],
         legend=:topleft,
         title=L"error with $N = %$N$ and $\sigma_w, \sigma_z = %$σwz$",
        )
savefig(p, "plots/stability.pdf")
