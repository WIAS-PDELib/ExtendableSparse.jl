"""
    ExtendableSparse

$(read(joinpath(@__DIR__, "..", "README.md"), String))
"""
module ExtendableSparse

using DocStringExtensions: DocStringExtensions, SIGNATURES, TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES
using ILUZero: ILUZero
using LinearAlgebra: LinearAlgebra, AbstractVecOrMat, Diagonal, Hermitian, Symmetric, Tridiagonal, convert, mul!, ldiv!
using SparseArrays: SparseArrays, AbstractSparseMatrix, AbstractSparseMatrixCSC, SparseMatrixCSC
using SparseArrays: dropzeros!, findnz, nzrange, sparse, spzeros, rowvals, getcolptr, nonzeros, nnz
using Sparspak: sparspaklu
using SciMLPublic: @public
using StaticArrays: StaticArrays, SMatrix, SVector


# Define our own constant here in order to be able to
# test things at least a little bit..
const USE_GPL_LIBS = Base.USE_GPL_LIBS


include("sparsematrixcsc.jl")
include("abstractsparsematrixextension.jl")
include("sparsematrixlnk.jl")
include("sparsematrixdilnkc.jl")
include("sparsematrixdict.jl")
include("abstractextendablesparsematrixcsc.jl")
include("genericmtextendablesparsematrixcsc.jl")
include("genericextendablesparsematrixcsc.jl")
include("typealiases.jl")


@public AbstractExtendableSparseMatrixCSC, AbstractSparseMatrixExtension
@public GenericExtendableSparseMatrixCSC, GenericMTExtendableSparseMatrixCSC
export MTExtendableSparseMatrixCSC, STExtendableSparseMatrixCSC
export ExtendableSparseMatrix, ExtendableSparseMatrixCSC
export flush!, updateindex!, rawupdateindex!, reset!, nnznew
@public partitioning!

export eliminate_dirichlet, eliminate_dirichlet!, mark_dirichlet


include("preconbuilders.jl")
export LinearSolvePreconBuilder, BlockPreconBuilder, JacobiPreconBuilder
@public ILUZeroPreconBuilder, ILUTPreconBuilder


include("sprand.jl")
export fdrand, fdrand!
@public sprand!, sprand_sdd!, fdrand_coo

end # module
