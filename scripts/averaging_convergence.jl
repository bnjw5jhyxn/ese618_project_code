include("../src/util.jl")
include("../src/baselines.jl")
using .Util: synthetic_system, simulate, make_ubar, true_markov_params, make_hankel, ho_kalman
using .Baselines: averaging, ordinary_least_squares
using LinearAlgebra: opnorm, I
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
μ = 1e-2
N = 10000
trials_per_ρ = 100
@assert T * L + 1 ≤ N

function ho_kalman_cycle(Gh::Matrix{Float64})::Matrix{Float64}
    Ah, Bh, Ch = ho_kalman(H=make_hankel(G=Gh, p=p, T1=T1, T2=T2),
                           n=n, m=m, p=p, T1=T1, T2=T2)
    Dh = Gh[:, 1:p]
    true_markov_params(A=Ah, B=Bh, C=Ch, D=Dh, T=T)
end

function averaging_error(ρ::Float64)::Vector{Float64}
    A, B, C, D = synthetic_system(n=n, p=p, m=m, ρ=ρ)
    Y, U = simulate(A=A, B=B, C=C, D=D, σu=σu, σw=σwz, σz=σwz, T=N)
    avg = averaging([(Y, U)], σu2=σu^2, T=T)
    ols = ordinary_least_squares(Y, U, T=T)
    G = true_markov_params(A=A, B=B, C=C, D=D, T=T)
    [
     opnorm(avg - ols),
     opnorm(ols - G),
     opnorm(avg - G),
    ]
end

xs = LinRange(1.0, 4.0, 200)
ρs = 1 .- 10 .^ (-xs)
errs = transpose(hcat([mean(hcat([averaging_error(ρ)
                                  for _ = 1:trials_per_ρ]...),
                           dims=2)
                       for ρ = ρs]...))
gr()
p = plot(
         xs, log10.(errs),
         xlabel=L"\rho(A) = 1 - 10^{-x}",
         ylabel=L"\log_{10}\Vert M \Vert _{\rm op}",
         label=[L"\hat G_{\rm avg} - \hat G_{\rm ols}" L"\hat G_{\rm ols} - G" L"\hat G_{\rm avg} - G"],
         legend=:topleft,
         title=L"N = %$N, \sigma_w, \sigma_z = %$σwz",
        )
savefig(p, "plots/averaging_convergence.pdf")
