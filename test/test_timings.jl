module test_timings
using Test
using ExtendableSparse
using ExtendableSparse: SparseMatrixLNK
using SparseArrays
using Random
using MultiFloats
using ForwardDiff
using BenchmarkTools
using Printf
using ExtendableSparse: GenericExtendableSparseMatrixCSC
using ExtendableSparse: SparseMatrixLNK, SparseMatrixDILNKC, SparseMatrixDict

const Dual64 = ForwardDiff.Dual{Float64, Float64, 1}

function test(Tm, T, k, l, m)
    t1 = @belapsed fdrand($T, $k, $l, $m, matrixtype = $SparseMatrixCSC) seconds = 0.1
    t2 = @belapsed fdrand($T, $k, $l, $m, matrixtype = $GenericExtendableSparseMatrixCSC{$(Tm)}) seconds = 0.1
    t3 = @belapsed fdrand($T, $k, $l, $m, matrixtype = $Tm) seconds = 0.1
    @printf(
        "%s (%d,%d,%d): CSC %.4f  Extendable %.4f  Extension %.4f\n",
        string(T),
        k,
        l,
        m,
        t1 * 1000,
        t2 * 1000,
        t3 * 1000
    )

    if !(t3 < t2 < t1)
        @warn """timing test failed for $T $k x $l x $m.
        If this occurs just once or twice, it is probably due to CPU noise.
        So we nevertheless count this as passing.
        """
    end
    return true
end

for Tm in [SparseMatrixLNK, SparseMatrixDict, SparseMatrixDILNKC]
    @show Tm
    @test test(Tm, Float64, 1000, 1, 1)
    @test test(Tm, Float64, 100, 100, 1)
    @test test(Tm, Float64, 20, 20, 20)

    @test test(Tm, Float64x2, 1000, 1, 1)
    @test test(Tm, Float64x2, 100, 100, 1)
    @test test(Tm, Float64x2, 20, 20, 20)

    @test test(Tm, Dual64, 1000, 1, 1)
    @test test(Tm, Dual64, 100, 100, 1)
    @test test(Tm, Dual64, 20, 20, 20)
end
end
