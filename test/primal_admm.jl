include("../src/util.jl")
include("../src/rank.jl")
include("../src/sysid.jl")
using .Util: synthetic_system, simulate, true_markov_params, make_hankel, ho_kalman
using .Sysid: ordinary_primal_admm, ordinary_dual_admm
using LinearAlgebra: opnorm
using Statistics: mean
using Plots: plot, savefig, gr
using LaTeXStrings: @L_str

n = 5
p = 3
m = 2
ρ = 0.9

σu = 1.0
σwz = 0.25

T1 = 9
T2 = 8
T = T1 + T2 + 1
L = 9
μs = [1e2, 1e3, 1e4, 1e5]
N0 = 200
trials_per_n = 1
@assert T * L + 1 ≤ 170

function ho_kalman_error(;
        A::Matrix{Float64},
        B::Matrix{Float64},
        C::Matrix{Float64},
        D::Matrix{Float64},
        Gh::Matrix{Float64},
    )::Float64
    G = true_markov_params(A=A, B=B, C=C, D=D, T=T)
    Ah, Bh, Ch = ho_kalman(H=make_hankel(G=G, p=p, T1=T1, T2=T2),
                           n=n, m=m, p=p, T1=T1, T2=T2)
    Dh = Gh[:, 1:p]
    opnorm(true_markov_params(A=Ah, B=Bh, C=Ch, D=Dh, T=T) - G)
end

function experiment_error(N::Int64, μs::Vector{Float64})::Vector{Float64}
    A, B, C, D = synthetic_system(n=n, p=p, m=m, ρ=ρ)
    Y, U = simulate(A=A, B=B, C=C, D=D, σu=σu, σw=σwz, σz=σwz, T=N)
    [ho_kalman_error(A=A, B=B, C=C, D=D,
                     Gh=ordinary_dual_admm(Y, U, T1=T1, T2=T2, μ=μ))
     for μ = μs]
end

Ns = 5000 : 5000 : 20000
errs = transpose(hcat([mean(hcat([experiment_error(N, μs)
                                  for _ = 1:trials_per_n]...),
                           dims=2)
                       for N = Ns]...))
gr()
p = plot(
         Ns, log10.(errs),
         xlabel=L"N",
         ylabel=L"\log_{10} \Vert \hat G - G \Vert _{\rm op}",
         label=[L"\mu = 10^2" L"\mu = 10^3" L"\mu = 10^4" L"\mu = 10^5"],
         title=L"error with $\rho(A) = %$ρ$ and $\sigma_w, \sigma_z = %$σwz$",
        )
savefig(p, "plots/primal_admm_test.pdf")
