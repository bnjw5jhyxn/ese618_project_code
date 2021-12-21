module Sysid

using LinearAlgebra: opnorm, I
using ..Util: make_ubar
using ..Rank: primal_admm, dual_admm

function ordinary_primal_admm(
        Y::Matrix{Float64},
        U::Matrix{Float64};
        T1::Int64, T2::Int64,
        μ::Float64,
    )::Matrix{Float64}
    @assert size(Y, 2) == size(U, 2)
    m = size(Y, 1)
    n = size(Y, 2)
    p = size(U, 1)
    T = T1 + T2 + 1
    Ubar = make_ubar(U, T)
    Ubart = transpose(Ubar)
    UUtInv = inv(Ubar * Ubart)
    primal_admm(A = x -> x * Ubar,
                Astar = x -> x * Ubart,
                AstarAinv = x -> x * UUtInv,
                opnormA = opnorm(Ubar),
                b = Y[:, T:end],
                𝚼 = 1.0 * Matrix(I, p * (T2 + 1), p * (T2 + 1)),
                m = m, n = p, j = T1 + 1, k = T2 + 1,
                μ = μ,
               )
end

function ordinary_dual_admm(
        Y::Matrix{Float64},
        U::Matrix{Float64};
        T1::Int64, T2::Int64,
        μ::Float64,
    )::Matrix{Float64}
    @assert size(Y, 2) == size(U, 2)
    m = size(Y, 1)
    n = size(Y, 2)
    p = size(U, 1)
    T = T1 + T2 + 1
    Ubar = make_ubar(U, T)
    Ubart = transpose(Ubar)
    UUtInv = inv(Ubar * Ubart)
    dual_admm(A = x -> x * Ubar,
                Astar = x -> x * Ubart,
                AstarAinv = x -> x * UUtInv,
                opnormA = opnorm(Ubar),
                b = Y[:, T:end],
                𝚼 = 1.0 * Matrix(I, p * (T2 + 1), p * (T2 + 1)),
                m = m, n = p, j = T1 + 1, k = T2 + 1,
                μ = μ,
               )
end

end
