include("../src/util.jl")
include("../src/baselines.jl")
using .Util: synthetic_system, simulate, true_markov_params
using .Baselines: averaging
using LinearAlgebra: norm

n = 5
p = 3
m = 2
ρ = 1.0

σu = 1.0
σw = 0.1
σz = 0.1

T1 = 6
T2 = 4
T = T1 + T2 + 1
total_timesteps = 100000
traj_length = 100000
num_trajs = div(total_timesteps, traj_length)

A, B, C, D = synthetic_system(n=n, p=p, m=m, ρ=ρ)
trajectories = [simulate(A=A, B=B, C=C, D=D,
                         σu=σu, σw=σw, σz=σz,
                         T=traj_length)
                for _ = 1:num_trajs]
G = true_markov_params(A=A, B=B, C=C, D=D, T=T)
Gh = averaging(trajectories, σu2=σu^2, T=T)
println(norm(G - Gh, Inf))
