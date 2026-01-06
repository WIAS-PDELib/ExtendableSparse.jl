module test_updates
using Test
using ExtendableSparse
using ExtendableSparse: GenericExtendableSparseMatrixCSC
using ExtendableSparse: SparseMatrixLNK, SparseMatrixDILNKC, SparseMatrixDict
using SparseArrays
using Random
using MultiFloats
using ForwardDiff
const Dual64 = ForwardDiff.Dual{Float64, Float64, 1}

function test(Tm, T)
    A = GenericExtendableSparseMatrixCSC{Tm}(T, 10, 10)
    @test nnz(A) == 0
    A[1, 3] = 5
    updateindex!(A, +, 6.0, 4, 5)
    updateindex!(A, +, 0.0, 2, 3)
    @test nnz(A) == 2
    rawupdateindex!(A, +, 0.0, 2, 3)
    @test nnz(A) == 3
    dropzeros!(A)
    @test nnz(A) == 2
    rawupdateindex!(A, +, 0.1, 2, 3)
    @test nnz(A) == 3
    dropzeros!(A)
    return @test nnz(A) == 3
end

for Tm in [SparseMatrixLNK, SparseMatrixDict, SparseMatrixDILNKC]
    test(Tm, Float64)
    test(Tm, Float64x2)
    test(Tm, Dual64)
end
end
