include("../src/util.jl")
include("../src/rank.jl")
include("../src/sysid.jl")
using .Util: synthetic_system, simulate, true_markov_params, make_hankel, ho_kalman
using .Sysid: ordinary_primal_admm, ordinary_dual_admm, prefiltered_dual_admm_fixedL
using LinearAlgebra: opnorm
using Statistics: mean
using Plots: plot, savefig, gr
using LaTeXStrings: @L_str

n = 5
p = 3
m = 2
ρ = 0.9999

σu = 1.0
σwz = 0.25

T1 = 9
T2 = 8
T = T1 + T2 + 1
L = 9
μ_pf = 1e-2
μs = [1e0, 1e1, 1e2, 1e3, 1e4]
N0 = 200
trials_per_n = 2
@assert T * L + 1 ≤ 170

function ho_kalman_error(;
        G::Matrix{Float64},
        Gh::Matrix{Float64},
    )::Float64
    Ah, Bh, Ch = ho_kalman(H=make_hankel(G=G, p=p, T1=T1, T2=T2),
                           n=n, m=m, p=p, T1=T1, T2=T2)
    Dh = Gh[:, 1:p]
    opnorm(true_markov_params(A=Ah, B=Bh, C=Ch, D=Dh, T=T) - G)
end

Ns = 5000 : 5000 : 20000
function experiment_error(μs::Vector{Float64})::Matrix{Float64}
    A, B, C, D = synthetic_system(n=n, p=p, m=m, ρ=ρ)
    Y, U = simulate(A=A, B=B, C=C, D=D, σu=σu, σw=σwz, σz=σwz, T=Ns[end])
    G = true_markov_params(A=A, B=B, C=C, D=D, T=T)
    err = zeros(size(Ns, 1), size(μs, 1))
    for j in 1:size(μs, 1)
        Gh = nothing
        Λh = nothing
        for i = 1:size(Ns, 1)
            Yn = Y[:, 1 : Ns[i]]
            Un = U[:, 1 : Ns[i]]
            #(Gh, Λh) = ordinary_dual_admm(Yn, Un, T1=T1, T2=T2, μ=μs[j],
            #                              G_init = Gh,
            #                              Λ_init = Λh)
            (Gh, Λh) = prefiltered_dual_admm_fixedL(Yn, Un, T1=T1, T2=T2, μ_nuc=μs[j],
                                                    μ_pf=μ_pf, L=L,
                                                    G_init = Gh,
                                                    Λ_init = Λh)
            err[i, j] = ho_kalman_error(G=G, Gh=Gh)
        end
    end
    err
end

errs = mean([experiment_error(μs) for _ = 1:trials_per_n])

gr()
p = plot(
         Ns, log10.(errs),
         xlabel=L"N",
         ylabel=L"\log_{10} \Vert \hat G - G \Vert _{\rm op}",
         label=[L"\mu = 10^0" L"\mu = 10^1" L"\mu = 10^2" L"\mu = 10^3" L"\mu = 10^4"],
         title=L"error with $\rho(A) = %$ρ$ and $\sigma_w, \sigma_z = %$σwz$",
        )
savefig(p, "plots/primal_admm_test.pdf")
