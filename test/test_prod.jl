module test_prod
using Test
using ExtendableSparse
using ExtendableSparse: ILUZeroPreconBuilder, IdentityPreconBuilder, ProductPreconBuilder
using IterativeSolvers
using LinearAlgebra
using Random

function main(; n = 100)
    seed = 1340
    Random.seed!(seed)
    A = fdrand(n, n)
    sol0 = ones(n^2)
    b = A * sol0

    # need to reinit random so the shadow residual is the same for all calls
    Random.seed!(seed)
    sol1, hist1 = bicgstabl(A, b, log = true)
    @test sol1 ≈ sol0 rtol = 1.0e-5
    @show hist1


    IdB = IdentityPreconBuilder()
    IluB = ILUZeroPreconBuilder()
    ProdB1 = ProductPreconBuilder(IdB, IluB)
    ProdB2 = ProductPreconBuilder(IluB, IdB)
    ProdB3 = ProductPreconBuilder(IluB, IluB)

    IdP, _ = IdB(A, 0)
    Random.seed!(seed)
    sol2, hist2 = bicgstabl(A, b, Pl = IdP, log = true)
    @test sol2 ≈ sol0  rtol = 1.0e-5
    @show hist2
    @test hist2.iters ≈ hist1.iters atol = 5

    IluP, _ = IluB(A, 0)
    Random.seed!(seed)
    sol3, hist3 = bicgstabl(A, b, Pl = IluP, log = true)
    @test sol3 ≈ sol0  rtol = 1.0e-5
    @show hist3
    @test hist3.iters < hist2.iters

    ProdP1, _ = ProdB1(A, 0)
    Random.seed!(seed)
    sol4, hist4 = bicgstabl(A, b, Pl = ProdP1, log = true)
    @show hist4
    @test sol4 ≈ sol0  rtol = 1.0e-5
    @test hist4.iters < hist3.iters

    ProdP2, _ = ProdB2(A, 0)
    Random.seed!(seed)
    sol5, hist5 = bicgstabl(A, b, Pl = ProdP2, log = true)
    @show hist5
    @test sol5 ≈ sol0  rtol = 1.0e-5
    @test hist5.iters ≈ hist4.iters atol = 7

    ProdP3, _ = ProdB3(A, 0)
    Random.seed!(seed)
    sol6, hist6 = bicgstabl(A, b, Pl = ProdP3, log = true)
    @show hist6
    @test sol6 ≈ sol0  rtol = 1.0e-5
    @test hist6.iters < hist5.iters


    return
end

main()


end
