include("../src/util.jl")
include("../src/baselines.jl")
include("../src/rank.jl")
include("../src/sysid.jl")
using .Util: synthetic_system, simulate, true_markov_params, make_hankel, ho_kalman
using .Baselines: averaging, ordinary_least_squares, prefiltered_least_squares_fixedL
using .Sysid: ordinary_primal_admm, ordinary_dual_admm, prefiltered_dual_admm_fixedL
using LinearAlgebra: opnorm
using Statistics: mean
using Plots: plot, savefig, gr
using LaTeXStrings: @L_str

n = 5
p = 3
m = 2
#ρ = 0.9
ρ = 0.9999
#ρ = 1.0001
#ρ = 1.001

σu = 1.0
σwz = 0.25

T1 = 9
T2 = 8
T = T1 + T2 + 1
L = 9
μ_pf = 1e-2
μs = [1e0, 1e1, 1e2, 1e3, 1e4]
N0 = 200
trials_per_n = 10
@assert T * L + 1 ≤ 170

Ns = 2000 : 2000 : 20000
function experiment_error()::Matrix{Float64}
    A, B, C, D = synthetic_system(n=n, p=p, m=m, ρ=ρ)
    Y, U = simulate(A=A, B=B, C=C, D=D, σu=σu, σw=σwz, σz=σwz, T=Ns[end])
    G = true_markov_params(A=A, B=B, C=C, D=D, T=T)
    err = zeros(size(Ns, 1), size(μs, 1) + 1)
    for j in 1:size(μs, 1)
        Gh = nothing
        Λh = nothing
        for i = 1:size(Ns, 1)
            Yn = Y[:, 1 : Ns[i]]
            Un = U[:, 1 : Ns[i]]
            (Gh, Λh) = prefiltered_dual_admm_fixedL(Yn, Un, T1=T1, T2=T2, μ_nuc=μs[j],
                                                    μ_pf=μ_pf, L=L,
                                                    G_init = Gh,
                                                    Λ_init = Λh)
            err[i, j] = opnorm(Gh - G)
        end
    end
    for i = 1:size(Ns, 1)
        Yn = Y[:, 1 : Ns[i]]
        Un = U[:, 1 : Ns[i]]
        err[i, size(μs, 1) + 1] = opnorm(prefiltered_least_squares_fixedL(Yn, Un, T=T,
                                                                          L=L, μ=μ_pf)
                                         - G)
    end
    err
end

errs = mean([experiment_error() for _ = 1:trials_per_n])

gr()
p = plot(
         Ns, log10.(errs),
         xlabel=L"N",
         ylabel=L"\log_{10} \Vert \hat G - G \Vert _{\rm op}",
         label=[L"\mu = 10^0" L"\mu = 10^1" L"\mu = 10^2" L"\mu = 10^3" L"\mu = 10^4" "PLS"],
         title=L"error with $\rho(A) = %$ρ$ and $\sigma_w, \sigma_z = %$σwz$",
        )
#savefig(p, "plots/prefiltered_rho_small.pdf")
savefig(p, "plots/prefiltered_rho_1me.pdf")
#savefig(p, "plots/prefiltered_rho_1pe.pdf")
#savefig(p, "plots/prefiltered_rho_big.pdf")
