module Baselines

using LinearAlgebra: transpose, I
using ..Util: make_ubar, make_k

"""
uses averaging to estimate Markov paremeters
from N independent trajectories, each of length T

the input is a sequence of pairs (Yn, Un),
where each column of Yn is an observation
and each column of Un is a control input
"""
function averaging(
        trajectories::Vector{Tuple{Matrix{Float64}, Matrix{Float64}}};
        σu2::Float64, T::Int64,
    )::Matrix{Float64}
    Y1, U1 = trajectories[1]
    m = size(Y1, 1)
    p = size(U1, 1)
    for (Yn, Un) = trajectories
        @assert size(Yn, 1) == m
        @assert size(Un, 1) == p
        @assert size(Yn, 2) == size(Un, 2)
    end
    Y = hcat([Yn[:, T:end] for (Yn, _) = trajectories]...)
    U = hcat([make_ubar(Un, T) for (_, Un) = trajectories]...)
    Y * transpose(U) / (size(Y, 2) * σu2)
end

"""
least-squares estimate
from Oymak and Ozay, 2019
"""
function ordinary_least_squares(
        Y::Matrix{Float64}, U::Matrix{Float64};
        T::Int64
    )::Matrix{Float64}
    @assert size(Y, 2) == size(U, 2)
    Y[:, T:end] / make_ubar(U, T)
end

"""
prefiltered least squares
from Simchowitz, Boczar, and Recht, 2019
"""
function prefiltered_least_squares_fixedL(
        Y::Matrix{Float64}, U::Matrix{Float64};
        T::Int64, L::Int64, μ::Float64,
    )::Matrix{Float64}
    K = make_k(Y, T=T, L=L)
    Kt = transpose(K)
    TLp1 = T * L + 1
    Ytail = Y[:, TLp1 : end]
    φ = Ytail * Kt / (K * Kt + 2μ^2 * I)
    (Ytail - φ * K) / make_ubar(U, T, Tf=TLp1)
end

end
