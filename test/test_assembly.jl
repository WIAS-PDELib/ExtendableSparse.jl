module test_assembly
using Test
using SparseArrays
using ExtendableSparse
using ExtendableSparse: GenericExtendableSparseMatrixCSC
using ExtendableSparse: SparseMatrixLNK, SparseMatrixDILNKC, SparseMatrixDict

function test(Tm; m = 1000, n = 1000, xnnz = 5000, nsplice = 1)
    A = GenericExtendableSparseMatrixCSC{Tm}{Float64, Int64}(m, n)
    m, n = size(A)
    S = spzeros(m, n)
    for isplice in 1:nsplice
        for inz in 1:xnnz
            i = rand((1:m))
            j = rand((1:n))
            a = 1.0 + rand()
            S[i, j] += a
            A[i, j] += a
        end
        flush!(A)
        for j in 1:n
            @assert(issorted(A.cscmatrix.rowval[A.cscmatrix.colptr[j]:(A.cscmatrix.colptr[j + 1] - 1)]))
        end
        @assert(nnz(S) == nnz(A))

        (I, J, V) = findnz(S)
        for inz in 1:nnz(S)
            @assert(A[I[inz], J[inz]] == V[inz])
        end

        (I, J, V) = findnz(A)
        for inz in 1:nnz(A)
            @assert(S[I[inz], J[inz]] == V[inz])
        end
    end
    return true
end
for Tm in [SparseMatrixLNK, SparseMatrixDict, SparseMatrixDILNKC]
    @show Tm
    @test test(Tm; m = 10, n = 10, xnnz = 5)
    @test test(Tm; m = 100, n = 100, xnnz = 500, nsplice = 2)
    @test test(Tm; m = 1000, n = 1000, xnnz = 5000, nsplice = 3)

    @test test(Tm; m = 20, n = 10, xnnz = 5)
    @test test(Tm; m = 200, n = 100, xnnz = 500, nsplice = 2)
    @test test(Tm; m = 2000, n = 1000, xnnz = 5000, nsplice = 3)

    @test test(Tm; m = 10, n = 20, xnnz = 5)
    @test test(Tm; m = 100, n = 200, xnnz = 500, nsplice = 2)
    @test test(Tm; m = 1000, n = 2000, xnnz = 5000, nsplice = 3)

    for irun in 1:10
        m = rand((1:10000))
        n = rand((1:10000))
        nnz = rand((1:10000))
        nsplice = rand((1:5))
        @test test(Tm; m = m, n = n, xnnz = nnz, nsplice = nsplice)
    end

end

end
