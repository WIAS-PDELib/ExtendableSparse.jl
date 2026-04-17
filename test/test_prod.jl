module test_prod
using Test
using ExtendableSparse
using ExtendableSparse: ILUZeroPreconBuilder, IdentityPreconBuilder, ProductPreconBuilder
using IterativeSolvers
using LinearAlgebra
using Random

function main(; n = 100)
    Random.seed!(2026)
    A = fdrand(n, n)
    sol0 = ones(n^2)
    b = A * sol0


    sol1, hist1 = bicgstabl(A, b, log = true)
    @test sol1 ≈ sol0 rtol = 1.0e-5
    @show hist1


    IdB = IdentityPreconBuilder()
    IluB = ILUZeroPreconBuilder()
    ProdB1 = ProductPreconBuilder(IdB, IluB)
    ProdB2 = ProductPreconBuilder(IluB, IdB)
    ProdB3 = ProductPreconBuilder(IluB, IluB)

    IdP, _ = IdB(A, 0)
    sol2, hist2 = bicgstabl(A, b, Pl = IdP, log = true)
    @test sol2 ≈ sol0  rtol = 1.0e-5
    @show hist2

    IluP, _ = IluB(A, 0)
    sol3, hist3 = bicgstabl(A, b, Pl = IluP, log = true)
    @test sol3 ≈ sol0
    @show hist3

    ProdP1, _ = ProdB1(A, 0)
    sol4, hist4 = bicgstabl(A, b, Pl = ProdP1, log = true)
    @show hist4
    @test sol4 ≈ sol0

    ProdP2, _ = ProdB2(A, 0)
    sol5, hist5 = bicgstabl(A, b, Pl = ProdP2, log = true)
    @show hist5
    @test sol5 ≈ sol0

    ProdP3, _ = ProdB3(A, 0)
    sol6, hist6 = bicgstabl(A, b, Pl = ProdP3, log = true)
    @show hist6
    @test sol6 ≈ sol0


    return
end

main()


end
