module test_block
using Test
using ExtendableSparse
using ExtendableSparse: BlockPreconditioner, JacobiPreconditioner
using ILUZero
using IterativeSolvers
using LinearAlgebra
using Sparspak
using AMGCLWrap

ExtendableSparse.allow_views(::typeof(ilu0)) = true

function main(; n = 100, blocksize = 4)

    A = fdrand(n, n)
    N = n^2

    partitioning = [i:blocksize:(n^2) for i in 1:blocksize ]
    sol0 = ones(N)

    b = A * ones(N)
    sol = cg(A, b, Pl = ilu0(A))

    @test sol ≈ sol0

    sol, hist = cg(A, b, Pl = BlockPreconditioner(A; partitioning, factorizations = JacobiPreconditioner), log = true)
    @test sol ≈ sol0

    sol, hist = cg(A, b, Pl = BlockPreconditioner(A; partitioning, factorizations = ilu0), log = true)
    @test sol ≈ sol0

    sol, hist = cg(A, b, Pl = BlockPreconditioner(A; partitioning, factorizations = sparspaklu), log = true)
    @test sol ≈ sol0

    partitioning = [i:(i + blocksize - 1) for i in 1:blocksize:N]
    sol, hist_eq = cg(A, b, Pl = BlockPreconditioner(A; partitioning, factorizations = sparspaklu), log = true)
    @test sol ≈ sol0

    sol, hist_pt = cg(A, b, Pl = JacobiPreconditioner(A; blocksize), log = true)
    @test sol ≈ sol0
    @test hist_pt.iters == hist_eq.iters


    return
end

main(n = 100)

end
