include("../src/util.jl")
include("../src/baselines.jl")
using .Util: synthetic_system, simulate, true_markov_params, make_ubar
using .Baselines: averaging, ordinary_least_squares
using LinearAlgebra: opnorm, pinv
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
μ = 1e-2
N0 = 100
trials_per_n = 20
@assert T ≤ N0

function avg_ols_diff(N::Int64)::Vector{Float64}
    A, B, C, D = synthetic_system(n=n, p=p, m=m, ρ=ρ)
    Y, U = simulate(A=A, B=B, C=C, D=D, σu=σu, σw=σwz, σz=σwz, T=N)
    G = true_markov_params(A=A, B=B, C=C, D=D, T=T)
    Ubar = make_ubar(U, T)
    [
     opnorm(averaging([(Y, U)], σu2=σu^2, T=T)
            - ordinary_least_squares(Y, U, T=T)),
     opnorm(pinv(Ubar) - transpose(Ubar) / (n * σu^2)),
     opnorm(Y),
    ]
end

Ns = N0 : 100 : 20000
errs = transpose(hcat([mean(hcat([avg_ols_diff(N)
                                  for _ = 1:trials_per_n]...),
                           dims=2)
                       for N = Ns]...))
gr()
p = plot(
         Ns, errs,
         xlabel=L"N",
         yscale=:log10,
         label=[L"\Vert \hat G_{\rm avg} - \hat G_{\rm OLS} \Vert _{\rm op}" L"\Vert U^+ - \frac{1}{n \sigma_u^2} U^\top \Vert_{\rm op}" L"\Vert Y \Vert_{\rm op}"],
         title="difference between averaging and OLS",
        )
savefig(p, "plots/avg_ols_diff.pdf")
