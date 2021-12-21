module Rank

using LinearAlgebra: svd, opnorm, norm, Diagonal
#using Printf: @printf
using ..Util: schatten_norm, frob_inpr, make_unclipped_hankel, hankel_adj

#=
this code is a port of the Matlab code accompanying the paper
"Hankel matrix rank minimization
with applications to system identification and realization"
by Maryam Fazel, Ting Kei Pong, Defeng Sun, and Paul Tseng
=#

function primal_admm(;
        A::Function,
        Astar::Function,
        AstarAinv::Function,
        opnormA::Float64,
        b::Matrix{Float64},
        𝚼::Matrix{Float64},
        m::Int64, n::Int64, j::Int64, k::Int64,
        μ::Float64,
        β::Float64 = 1.0,
        #β::Float64 = 0.5 * μ * min(j,k) / opnorm(b),
        τ::Float64 = 1.61,
        maxIter::Int64 = 10000,
        gap_check_freq::Int64 = 10,
        tolerance::Float64 = 1e-4,
        ε::Float64 = 1e-8,
    )::Matrix{Float64}
    r = min(j, k)
    σ = β / (β * r + opnormA^2)
    (p1, p2) = size(b)
    (nk, q) = size(𝚼)
    @assert nk == n * k
    @assert opnorm(𝚼) ≤ 1

    function H(X::Matrix{Float64})::Matrix{Float64}
        make_unclipped_hankel(X = X, n = n, j = j, k = k) * 𝚼
    end
    function Hstar(X::Matrix{Float64})::Matrix{Float64}
        hankel_adj(W = X * transpose(𝚼), m = m, n = n)
    end
    function f_obj(X::Matrix{Float64})::Float64
        0.5 * norm(A(X) - b)^2 + μ * schatten_norm(H(X))
    end
    function d2(X::Matrix{Float64})::Tuple{Matrix{Float64}, Float64}
        HstarX = Hstar(X)
        RX = AstarAinv(HstarX)
        Astarb = Astar(b)
        bbar = AstarAinv(Astarb)
        (RX + bbar,
         0.5 * (frob_inpr(HstarX, RX)
                + 2 * frob_inpr(HstarX, bbar)
                + frob_inpr(Astarb, bbar)
                - norm(b)^2))
    end

    Y = zeros(m * j, q)
    y = zeros(m, n * (j + k - 1))
    Λdβ = zeros(m * j, q)
    Y_prev = ones(m * j, q)
    y_change = ones(m, n * (j + k - 1))
    Λdβ_change = ones(m * j, q)
    numIter = 0
    min_fval = 1e20
    min_dval = 1e20
    y_best = zeros(m, n * (j + k - 1))
    duality_gap = 1.0
    while (numIter < maxIter && duality_gap ≥ tolerance
           && (norm(y_change) ≥ ε || norm(Λdβ_change) ≥ ε
               || norm(Y - Y_prev) ≥ ε))
        F = svd(-H(y) + Λdβ)
        Y_prev = Y
        Y = F.U * Diagonal(max.(F.S .- μ/β, 0)) * F.Vt
        y_change = σ * (Hstar(-Λdβ + H(y) + Y) + Astar(A(y) - b) / β)
        Λdβ_change = τ * (Y + H(y))
        y -= y_change
        Λdβ -= Λdβ_change
        numIter += 1
        if (numIter % gap_check_freq == 0
            || (norm(y_change) < ε && norm(Λdβ_change) < ε
                && norm(Y - Y_prev) < ε))
            FΛ = svd(Λdβ * β)
            PΛ = FΛ.U * Diagonal(min.(F.S, μ)) * FΛ.Vt
            fval = f_obj(y)
            (yh, dval) = d2(PΛ)
            fvalh = f_obj(yh)
            if fval < min_fval
                min_fval = fval
                y_best = y
            end
            if fvalh < min_fval
                min_fval = fvalh
                y_best = yh
            end
            min_dval = min(min_dval, dval)
            duality_gap = (min_fval + min_dval) / max(1, abs(min_dval))
            #@printf "fval = %f, dval = %f\n" fval dval
        end
    end

    #@printf "μ = %f, duality_gap = %f, numIter = %d\n" μ duality_gap numIter
    y_best
end

function dual_admm(;
        A::Function,
        Astar::Function,
        AstarAinv::Function,
        opnormA::Float64,
        b::Matrix{Float64},
        𝚼::Matrix{Float64},
        m::Int64, n::Int64, j::Int64, k::Int64,
        μ::Float64,
        #β::Float64 = 1.0,
        β::Float64 = opnorm(b) / (16 * μ * min(j,k)),
        τ::Float64 = 1.61,
        maxIter::Int64 = 10000,
        gap_check_freq::Int64 = 10,
        tolerance::Float64 = 1e-4,
        ε::Float64 = 1e-8,
    )::Matrix{Float64}
    r = min(j, k)
    σ1 = 1 / opnormA^2
    σ2 = 1 / r
    (p1, p2) = size(b)
    (nk, q) = size(𝚼)
    @assert nk == n * k
    @assert opnorm(𝚼) ≤ 1

    function H(X::Matrix{Float64})::Matrix{Float64}
        make_unclipped_hankel(X = X, n = n, j = j, k = k) * 𝚼
    end
    function Hstar(X::Matrix{Float64})::Matrix{Float64}
        hankel_adj(W = X * transpose(𝚼), m = m, n = n)
    end
    function f_obj(X::Matrix{Float64})::Float64
        0.5 * norm(A(X) - b)^2 + μ * schatten_norm(H(X))
    end
    function d2(X::Matrix{Float64})::Tuple{Matrix{Float64}, Float64}
        HstarX = Hstar(X)
        RX = AstarAinv(HstarX)
        Astarb = Astar(b)
        bbar = AstarAinv(Astarb)
        (RX + bbar,
         0.5 * (frob_inpr(HstarX, RX)
                + 2 * frob_inpr(HstarX, bbar)
                + frob_inpr(Astarb, bbar)
                - norm(b)^2))
    end

    σdσpβ = σ1 / (σ1 + β)
    βdσ = β / σ1
    τβ = τ * β

    𝛄 = zeros(p1, p2)
    Λ = zeros(m * j, q)
    y = zeros(m, n * (j + k - 1))
    𝛄_prev = ones(p1, p2)
    Λ_prev = ones(m * j, q)
    y_change = ones(m, n * (j + k - 1))

    numIter = 0
    min_fval = 1e20
    min_dval = 1e20
    y_best = zeros(m, n * (j + k - 1))
    duality_gap = 1.0
    while (numIter < maxIter && duality_gap ≥ tolerance
           && (norm(𝛄 - 𝛄_prev) ≥ ε
               || norm(Λ - Λ_prev) ≥ ε
               || norm(y_change) ≥ ε))
        𝛄 = σdσpβ * (b + βdσ * 𝛄 - A(y) - β * A(Hstar(Λ) + Astar(𝛄)))
        F = svd(Λ - σ2 * (H(y) / β + H(Hstar(Λ) + Astar(𝛄))))
        Λ = F.U * Diagonal(min.(F.S, μ)) * F.Vt
        y_change = τβ * (Hstar(Λ) + Astar(𝛄))
        y += y_change
        numIter += 1
        if (numIter % gap_check_freq == 0
            || (norm(𝛄 - 𝛄_prev) < ε
                && norm(Λ - Λ_prev) < ε
                && norm(y_change) < ε))
            fval = f_obj(y)
            (yh, dval) = d2(Λ)
            fvalh = f_obj(yh)
            if fval < min_fval
                min_fval = fval
                y_best = y
            end
            if fvalh < min_fval
                min_fval = fvalh
                y_best = yh
            end
            min_dval = min(min_dval, dval)
            duality_gap = (min_fval + min_dval) / max(1, abs(min_dval))
            #@printf "fval = %f, dval = %f\n" fval dval
        end
    end

    #@printf "μ = %f, duality_gap = %f, numIter = %d\n" μ duality_gap numIter
    y_best
end

end
