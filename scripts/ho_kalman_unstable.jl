include("../src/util.jl")
using .Util: synthetic_system, true_markov_params, ho_kalman, make_hankel
using LinearAlgebra: opnorm
using Statistics: mean
using Plots: plot, savefig, gr
using LaTeXStrings: @L_str

n = 5
p = 3
m = 2

T1 = 45
T2 = 40
T = T1 + T2 + 1
num_trials = 100

function rand_sys_err(ρ::Float64)::Float64
    A, B, C, D = synthetic_system(n=n, p=p, m=m, ρ=ρ)
    G = true_markov_params(A=A, B=B, C=C, D=D, T=T)
    Ah, Bh, Ch = ho_kalman(H=make_hankel(G=G, p=p, T1=T1, T2=T2),
                           n=n, m=m, p=p, T1=T1, T2=T2)
    opnorm(true_markov_params(A=Ah, B=Bh, C=Ch, D=D, T=T) - G)
end

ρs = LinRange(0.5, 1.5, 100)
errs = [mean([rand_sys_err(ρ) for _ = 1:num_trials]) for ρ = ρs]
gr()
p = plot(
         ρs, errs,
         xlabel=L"\rho(A)",
         ylabel=L"\Vert \hat G - G \Vert _{\rm op}",
         yscale=:log10,
         label="",
         title="error of Ho-Kalman algorithm for unstable systems",
        )
savefig(p, "plots/ho_kalman_unstable.svg")
