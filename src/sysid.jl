module Sysid

using LinearAlgebra: opnorm, I
using ..Util: make_ubar, make_k
using ..Rank: primal_admm, dual_admm

function ordinary_primal_admm(
        Y::Matrix{Float64},
        U::Matrix{Float64};
        T1::Int64, T2::Int64,
        μ::Float64,
        G_init::Union{Matrix{Float64}, Nothing} = nothing,
        Λ_init::Union{Matrix{Float64}, Nothing} = nothing,
    )::Tuple{Matrix{Float64}, Matrix{Float64}}
    @assert size(Y, 2) == size(U, 2)
    m = size(Y, 1)
    n = size(Y, 2)
    p = size(U, 1)
    T = T1 + T2 + 1
    Ubar = make_ubar(U, T)
    Ubart = transpose(Ubar)
    UUtInv = inv(Ubar * Ubart)
    (G, Λ, _, _) = primal_admm(A = x -> x * Ubar,
                               Astar = x -> x * Ubart,
                               AstarAinv = x -> x * UUtInv,
                               opnormA = opnorm(Ubar),
                               b = Y[:, T:end],
                               𝚼 = 1.0 * Matrix(I, p * (T2 + 1), p * (T2 + 1)),
                               m = m, n = p, j = T1 + 1, k = T2 + 1,
                               μ = μ,
                               y_init = G_init,
                               Λ_init = Λ_init,
                              )
    (G, Λ)
end

function ordinary_dual_admm(
        Y::Matrix{Float64},
        U::Matrix{Float64};
        T1::Int64, T2::Int64,
        μ::Float64,
        G_init::Union{Matrix{Float64}, Nothing} = nothing,
        Λ_init::Union{Matrix{Float64}, Nothing} = nothing,
    )::Tuple{Matrix{Float64}, Matrix{Float64}}
    @assert size(Y, 2) == size(U, 2)
    m = size(Y, 1)
    n = size(Y, 2)
    p = size(U, 1)
    T = T1 + T2 + 1
    Ubar = make_ubar(U, T)
    Ubart = transpose(Ubar)
    UUtInv = inv(Ubar * Ubart)
    (G, Λ, _, _) = dual_admm(A = x -> x * Ubar,
                             Astar = x -> x * Ubart,
                             AstarAinv = x -> x * UUtInv,
                             opnormA = opnorm(Ubar),
                             b = Y[:, T:end],
                             𝚼 = 1.0 * Matrix(I, p * (T2 + 1), p * (T2 + 1)),
                             m = m, n = p, j = T1 + 1, k = T2 + 1,
                             μ = μ,
                             y_init = G_init,
                             Λ_init = Λ_init,
                            )
    (G, Λ)
end

function prefiltered_dual_admm_fixedL(
        Y::Matrix{Float64},
        U::Matrix{Float64};
        T1::Int64, T2::Int64,
        μ_nuc::Float64,
        L::Int64, μ_pf::Float64,
        G_init::Union{Matrix{Float64}, Nothing} = nothing,
        Λ_init::Union{Matrix{Float64}, Nothing} = nothing,
    )::Tuple{Matrix{Float64}, Matrix{Float64}}
    @assert size(Y, 2) == size(U, 2)
    m = size(Y, 1)
    n = size(Y, 2)
    p = size(U, 1)
    T = T1 + T2 + 1

    # prefilter
    K = make_k(Y, T=T, L=L)
    Kt = transpose(K)
    TLp1 = T * L + 1
    Ytail = Y[:, TLp1 : end]
    φ = Ytail * Kt / (K * Kt + 2μ_pf^2 * I)

    Ubar = make_ubar(U, T, Tf=TLp1)
    Ubart = transpose(Ubar)
    UUtInv = inv(Ubar * Ubart)
    (G, Λ, _, _) = dual_admm(A = x -> x * Ubar,
                             Astar = x -> x * Ubart,
                             AstarAinv = x -> x * UUtInv,
                             opnormA = opnorm(Ubar),
                             b = Ytail - φ * K,
                             𝚼 = 1.0 * Matrix(I, p * (T2 + 1), p * (T2 + 1)),
                             m = m, n = p, j = T1 + 1, k = T2 + 1,
                             μ = μ_nuc,
                             y_init = G_init,
                             Λ_init = Λ_init,
                            )
    (G, Λ)
end

end
