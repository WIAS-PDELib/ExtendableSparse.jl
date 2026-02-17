module test_block
using AMGCLWrap
using ExtendableSparse
using ExtendableSparse: BlockPreconditioner, jacobi, SchurComplementPreconBuilder
using ILUZero, AlgebraicMultigrid
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using Sparspak
using Test

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

    # Schur complement: create a saddle point system
    let
        m = n ÷ 10
        B = I[1:(n^2), 1:(m^2)]
        M = [ A B; B' spzeros(m^2, m^2)]

        sol1 = ones(n^2 + m^2)
        c = M * sol1

        sol = cg(M, c, Pl = SchurComplementPreconBuilder(n^2, lu)(M))
        @test sol ≈ sol1
    end

    return
end

main(n = 100)

end
