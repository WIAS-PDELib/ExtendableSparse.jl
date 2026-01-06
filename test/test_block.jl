module test_block
using Test
using ExtendableSparse
using ExtendableSparse: BlockPreconditioner, jacobi
using ILUZero, AlgebraicMultigrid
using IterativeSolvers
using LinearAlgebra
using Sparspak
using AMGCLWrap

ExtendableSparse.allow_views(::typeof(ilu0)) = true

function main(; n = 100)

    A = fdrand(n, n)
    partitioning = [1:2:(n^2), 2:2:(n^2)]
    sol0 = ones(n^2)
    b = A * ones(n^2)
    sol = cg(A, b, Pl = ilu0(A))

    @test sol ≈ sol0

    sol = cg(A, b, Pl = BlockPreconditioner(A; partitioning, factorization = ilu0))
    @test sol ≈ sol0

    sol = cg(A, b, Pl = BlockPreconditioner(A; partitioning, factorization = jacobi))
    @test sol ≈ sol0

    sol = cg(A, b, Pl = BlockPreconditioner(A; partitioning, factorization = sparspaklu))
    @test sol ≈ sol0

    return
end

main(n = 100)

end
