module Util

using LinearAlgebra: eigvals, svd, Diagonal

function spectral_radius(A::Matrix{Float64})::Float64
    maximum(abs, eigvals(A))
end

"""
takes a matrix U whose columns are u_t
and returns a matrix Ubar whose columns are the concatenation of
u_t, u_{t-1}, ..., u_{t-T+1}
"""
function make_ubar(U::Matrix{Float64}, T::Int64; Tf::Int64 = T)
    vcat([U[:, Tf-k : size(U,2)-k] for k = 0:T-1]...)
end

"""
takes a matrix Y whose columns are y_t
and returns a matrix K whose columns are the concatenation of
y_{t-T}, y_{t-2T}, ..., y_{t-LT}
"""
function make_k(Y::Matrix{Float64}; T::Int64, L::Int64)
    vcat([Y[:, T*L+1 - k*T : size(Y,2) - k*T] for k = 1:L]...)
end

"""
takes a m × p(T1 + T2 + 1) block matrix G
and outputs a m T1 × p(T2 + 1) block matrix H
such that the (i,j)th block of H is (i+j)th block of G
"""
function make_hankel(;
        G::Matrix{Float64},
        p::Int64, T1::Int64, T2::Int64,
    )::Matrix{Float64}
    vcat([G[:, i*p+1 : (i+T2+1)*p] for i = 1:T1]...)
end

function synthetic_system(;
        n::Int64, p::Int64, m::Int64, ρ::Float64,
    )::Tuple{Matrix{Float64},
             Matrix{Float64},
             Matrix{Float64},
             Matrix{Float64},
            }
    A_raw = randn(Float64, (n, n))
    (
     ρ * A_raw / spectral_radius(A_raw),
     randn(Float64, (n, p)) / sqrt(n),
     randn(Float64, (m, n)) / sqrt(m),
     randn(Float64, (m, p)) / sqrt(m),
    )
end

function true_markov_params(;
        A::Matrix{Float64}, B::Matrix{Float64},
        C::Matrix{Float64}, D::Matrix{Float64},
        T::Int64,
    )::Matrix{Float64}
    hcat(D, [C * A^k * B for k = 0:T-2]...)
end

function simulate(;
        A::Matrix{Float64}, B::Matrix{Float64},
        C::Matrix{Float64}, D::Matrix{Float64},
        σu::Float64, σw::Float64, σz::Float64,
        T::Int64,
    )::Tuple{Matrix{Float64}, Matrix{Float64}}
    n, n1 = size(A)
    n2, p = size(B)
    m, n3 = size(C)
    m1, p1 = size(D)
    @assert n == n1 == n2 == n3
    @assert m == m1
    @assert p == p1
    U = σu * randn(Float64, (p, T))
    W = σw * randn(Float64, (n, T))
    Z = σz * randn(Float64, (m, T))
    x = zeros(Float64, n)
    Y = zeros(Float64, (m, T))
    for t = 1:T
        Y[:,t] = C * x + D * U[:,t] + Z[:,t]
        x = A * x + B * U[:,t] + W[:,t]
    end
    Y, U
end

"""
Ho-Kalman algorithm
implementation copied from pseudocode in Oymak and Ozay, 2019
"""
function ho_kalman(;
        H::Matrix{Float64},
        n::Int64,
        m::Int64, p::Int64,
        T1::Int64, T2::Int64,
    )::Tuple{Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}
    Hm = H[:, 1 : p*T2]
    F = svd(Hm)
    n_eff = min(n, size(F.S, 1))
    U = F.U[:, 1:n_eff]
    Σh = Diagonal(sqrt.(F.S[1:n_eff]))
    Vt = F.Vt[1:n_eff, :]
    O = U * Σh
    Q = Σh * Vt
    Hp = H[:, p+1 : p*(T2+1)]
    O \ Hp / Q, Q[:, 1:p], O[1:m, :]
end

end
