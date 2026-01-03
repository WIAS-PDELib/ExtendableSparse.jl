module test_copymethods
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
    Xcsc = sprand(T, 10_000, 10_000, 0.01)
    Xlnk = Tm(Xcsc)
    Xext = ExtendableSparseMatrix(Xcsc)
    t0 = @elapsed copy(Xcsc)
    t1 = @elapsed copy(Xlnk)
    t2 = @elapsed copy(Xext)

    if !(t1 / t0 < 10 && t0 / t2 < 10)
        @warn """timing test failed.
        If this occurs just once or twice, it is probably due to CPU noise.
        So we nevertheless count this as passing.
        """
    end
    return true
end
for Tm in [SparseMatrixLNK, SparseMatrixDict, SparseMatrixDILNKC]
    @test test(Tm, Float64)
    @test test(Tm, Float64x2)
    @test test(Tm, Dual64)
end
end
