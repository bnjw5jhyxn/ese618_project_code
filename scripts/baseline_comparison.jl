include("../src/util.jl")
include("../src/baselines.jl")
include("../src/rank.jl")
include("../src/sysid.jl")
using .Util: synthetic_system, simulate, true_markov_params, make_hankel, ho_kalman
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
μ_pf = 1e-2
μ_nuc = 1e1
N0 = 200
trials_per_n = 30
@assert T * L + 1 ≤ 170

Ns = N0 : 100 : 20000
function experiment_error()::Matrix{Float64}
    A, B, C, D = synthetic_system(n=n, p=p, m=m, ρ=ρ)
    Y, U = simulate(A=A, B=B, C=C, D=D, σu=σu, σw=σwz, σz=σwz, T=Ns[end])
    G = true_markov_params(A=A, B=B, C=C, D=D, T=T)
    err = zeros(size(Ns, 1), 3)
    for i = 1:size(Ns, 1)
        Yn = Y[:, 1 : Ns[i]]
        Un = U[:, 1 : Ns[i]]
        err[i, 1] = opnorm(averaging([(Yn, Un)], σu2=σu^2, T=T) - G)
        err[i, 2] = opnorm(ordinary_least_squares(Yn, Un, T=T) - G)
        err[i, 3] = opnorm(prefiltered_least_squares_fixedL(Yn, Un, T=T, L=L, μ=μ_pf) - G)
    end
    err
end

errs = mean([experiment_error() for _ = 1:trials_per_n])

gr()
p = plot(
         Ns, log10.(errs),
         xlabel=L"N",
         ylabel=L"\log_{10} \Vert \hat G - G \Vert _{\rm op}",
         label=["avg" "OLS" "PLS"],
         title=L"error with $\rho(A) = %$ρ$ and $\sigma_w, \sigma_z = %$σwz$",
        )
#savefig(p, "plots/rho_small.pdf")
#savefig(p, "plots/rho_1me.pdf")
#savefig(p, "plots/rho_1pe.pdf")
savefig(p, "plots/rho_big.pdf")
