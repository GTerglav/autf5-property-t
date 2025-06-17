using Test
using LinearAlgebra
using SparseArrays

using Groups
import Groups.MatrixGroups

using PropertyT
import SymbolicWedderburn as SW
import StarAlgebras as SA
import PermutationGroups as PG

include("optimizers.jl")
include("check_positivity.jl")

N = 5
G = SpecialAutomorphismGroup(FreeGroup(N))
@info "running tests for" G
RG, S, sizes = PropertyT.group_algebra(G; halfradius = 2)

P = PG.PermGroup(PG.perm"(1,2)", PG.Perm(circshift(1:N, -1)))
Σ = Groups.Constructions.WreathProduct(PG.PermGroup(PG.perm"(1,2)"), P)
act = PropertyT.action_by_conjugation(G, Σ)
wd = SW.WedderburnDecomposition(
    Float64,
    Σ,
    act,
    SA.basis(RG),
    SA.Basis{UInt16}(@view SA.basis(RG)[1:sizes[2]]),
)
@info wd

Δ = let RG = RG, S = S
    RG(length(S)) - sum(RG(s) for s in S)
end

elt = Δ^2
unit = Δ
ub = Inf

status, certified, λ_cert = check_positivity(
    elt,
    unit,
    wd;
    upper_bound = ub,
    halfradius = 2,
    optimizer = cosmo_optimizer(;
        eps = 1e-7,
        max_iters = 10_000,
        accel = 50,
        alpha = 1.9,
    ),
)

@test status == JuMP.OPTIMAL
@test certified
@test λ_cert > 0