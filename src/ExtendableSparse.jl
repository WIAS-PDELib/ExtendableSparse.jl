module ExtendableSparse

using DocStringExtensions: DocStringExtensions, SIGNATURES, TYPEDEF, TYPEDFIELDS
using ILUZero: ILUZero, ldiv!, nnz
using LinearAlgebra: LinearAlgebra, Diagonal, Hermitian, Symmetric, Tridiagonal, convert, mul!, norm
using SparseArrays: SparseArrays, AbstractSparseMatrix, SparseMatrixCSC,
    dropzeros!, findnz, nzrange, sparse, spzeros
using Sparspak: sparspaklu
using StaticArrays: StaticArrays, SMatrix, SVector
import SparseArrays: AbstractSparseMatrixCSC, rowvals, getcolptr, nonzeros


# Define our own constant here in order to be able to
# test things at least a little bit..
const USE_GPL_LIBS = Base.USE_GPL_LIBS


include("compat.jl") # @public

include("matrix/sparsematrixcsc.jl")
include("matrix/abstractsparsematrixextension.jl")
include("matrix/sparsematrixlnk.jl")
include("matrix/sparsematrixdilnkc.jl")
include("matrix/abstractextendablesparsematrixcsc.jl")
include("matrix/genericmtextendablesparsematrixcsc.jl")
include("matrix/genericextendablesparsematrixcsc.jl")


"""
    MTExtendableSparseMatrixCSC

Multithreaded extendable sparse matrix  (Experimental).

Aliased to [`GenericMTExtendableSparseMatricCSC`](@ref) with [`SparseMatrixDILNKC`](@ref) 
scalar matrix parameter.
"""
const MTExtendableSparseMatrixCSC{Tv, Ti} = GenericMTExtendableSparseMatrixCSC{SparseMatrixDILNKC{Tv, Ti}, Tv, Ti}
MTExtendableSparseMatrixCSC(m, n, args...) = MTExtendableSparseMatrixCSC{Float64, Int64}(m, n, args...)

"""
    STExtendableSparseMatrixCSC

Single threaded extendable sparse matrix (Experimental).

Aliased to [`GenericExtendableSparseMatricCSC`](@ref) with [`SparseMatrixDILNKC`](@ref) 
scalar matrix parameter.
"""

const STExtendableSparseMatrixCSC{Tv, Ti} = GenericExtendableSparseMatrixCSC{SparseMatrixLNK{Tv, Ti}, Tv, Ti}
STExtendableSparseMatrixCSC(::Type{Tv}, m::Number, n::Number) where {Tv} = STExtendableSparseMatrixCSC{Tv, Int64}(m, n)
STExtendableSparseMatrixCSC(m::Number, n::Number) = STExtendableSparseMatrixCSC{Float64, Int64}(m, n)
function STExtendableSparseMatrixCSC(A::AbstractSparseMatrixCSC{Tv, Ti}) where {Tv, Ti <: Integer}
    return GenericExtendableSparseMatrixCSC(
        SparseMatrixCSC(A),
        SparseMatrixLNK{Tv, Ti}(size(A)...)
    )
end
STExtendableSparseMatrixCSC(D::Diagonal) = STExtendableSparseMatrixCSC(sparse(D))
STExtendableSparseMatrixCSC(I, J, V::AbstractVector) = STExtendableSparseMatrixCSC(sparse(I, J, V))
STExtendableSparseMatrixCSC(I, J, V::AbstractVector, m, n) = STExtendableSparseMatrixCSC(sparse(I, J, V, m, n))
STExtendableSparseMatrixCSC(I, J, V::AbstractVector, combine::Function) = STExtendableSparseMatrixCSC(sparse(I, J, V, combine))
STExtendableSparseMatrixCSC(I, J, V::AbstractVector, m, n, combine::Function) = STExtendableSparseMatrixCSC(sparse(I, J, V, m, n, combine))


const ExtendableSparseMatrixCSC = STExtendableSparseMatrixCSC
const ExtendableSparseMatrix = STExtendableSparseMatrixCSC

"""
    pointblock(matrix,blocksize)

Create a pointblock matrix.
"""
function pointblock(A0::ExtendableSparseMatrixCSC{Tv, Ti}, blocksize) where {Tv, Ti}
    A = SparseMatrixCSC(A0)
    colptr = A.colptr
    rowval = A.rowval
    nzval = A.nzval
    n = A.n
    block = zeros(Tv, blocksize, blocksize)
    nblock = n ÷ blocksize
    b = SMatrix{blocksize, blocksize}(block)
    Tb = typeof(b)
    Ab = ExtendableSparseMatrixCSC{Tb, Ti}(nblock, nblock)


    for i in 1:n
        for k in colptr[i]:(colptr[i + 1] - 1)
            j = rowval[k]
            iblock = (i - 1) ÷ blocksize + 1
            jblock = (j - 1) ÷ blocksize + 1
            ii = (i - 1) % blocksize + 1
            jj = (j - 1) % blocksize + 1
            block[ii, jj] = nzval[k]
            rawupdateindex!(Ab, +, SMatrix{blocksize, blocksize}(block), iblock, jblock)
            block[ii, jj] = zero(Tv)
        end
    end
    return flush!(Ab)
end


export ExtendableSparseMatrixCSC, MTExtendableSparseMatrixCSC, STExtendableSparseMatrixCSC, GenericMTExtendableSparseMatrixCSC
export SparseMatrixLNK, ExtendableSparseMatrix, flush!, nnz, updateindex!, rawupdateindex!, colptrs, sparse, reset!, nnznew
export partitioning!

export eliminate_dirichlet, eliminate_dirichlet!, mark_dirichlet


include("preconbuilders.jl")
export LinearSolvePreconBuilder, BlockPreconBuilder, JacobiPreconBuilder

@public ILUZeroPreconBuilder, ILUTPreconBuilder, SmoothedAggregationPreconBuilder, RugeStubenPreconBuilder


include("matrix/sprand.jl")
export sprand!, sprand_sdd!, fdrand, fdrand!, fdrand_coo, solverbenchmark

export rawupdateindex!, updateindex!


end # module
