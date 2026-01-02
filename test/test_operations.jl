module test_operations
using Test
using SparseArrays
using LinearAlgebra
using ExtendableSparse
using ExtendableSparse: sprand_sdd!, SparseMatrixLNK

#####################################################################
function test_addition(; m = 10, n = 10, d = 0.1)
    csc = sprand(m, n, d)
    lnk = SparseMatrixLNK(csc)
    csc2 = csc + lnk
    return csc2 == 2 * csc
end

function test_invert(n)
    A = ExtendableSparseMatrix(n, n)
    sprand_sdd!(A)
    b = rand(n)
    x = A \ b
    Ax = A * x
    return b ≈ Ax
end

function test()
    for irun in 1:10
        m = rand((1:1000))
        n = rand((1:1000))
        d = 0.3 * rand()
        @test test_addition(m = m, n = n, d = d)
    end

    @test test_invert(10)
    @test test_invert(100)
    @test test_invert(1000)

    A = sprand(10, 10, 0.1)
    B = sprand(10, 10, 0.1)
    extA = ExtendableSparseMatrix(A)
    extB = ExtendableSparseMatrix(B)

    @test A + extB == A + B
    @test A - extB == A - B
    @test extA + B == A + B
    @test extA - B == A - B
    @test extA * extB == ExtendableSparseMatrix(A * B)
    @test extA + extB == ExtendableSparseMatrix(A + B)
    @test extA - extB == ExtendableSparseMatrix(A - B)

    @test isa(extA * extB, ExtendableSparseMatrix)
    @test isa(extA + extB, ExtendableSparseMatrix)
    @test isa(extA - extB, ExtendableSparseMatrix)

    D = Diagonal(rand(10))
    @test isa(D * extA, ExtendableSparseMatrix)
    return @test isa(extA * D, ExtendableSparseMatrix)
end

test()
end
