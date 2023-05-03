module ExtendableSparse
using SparseArrays
using LinearAlgebra
using Sparspak
using ILUZero

# Define our own constant here in order to be able to
# test things at least a little bit..
const USE_GPL_LIBS = Base.USE_GPL_LIBS

if USE_GPL_LIBS
    using SuiteSparse
end

using Requires

using DocStringExtensions

import SparseArrays: AbstractSparseMatrixCSC, rowvals, getcolptr, nonzeros

include("matrix/sparsematrixcsc.jl")
include("matrix/sparsematrixlnk.jl")
include("matrix/extendable.jl")

export SparseMatrixLNK,
       ExtendableSparseMatrix, flush!, nnz, updateindex!, rawupdateindex!, colptrs, sparse

include("factorizations/factorizations.jl")

export JacobiPreconditioner,
    ILU0Preconditioner,
    ILUZeroPreconditioner,
    ParallelJacobiPreconditioner,
    ParallelILU0Preconditioner,
    BlockPreconditioner,allow_views,
    reorderlinsys

export AbstractFactorization, LUFactorization, CholeskyFactorization
export issolver
export factorize!, update!

include("factorizations/simple_iteration.jl")
export simple, simple!

include("matrix/sprand.jl")
export sprand!, sprand_sdd!, fdrand, fdrand!, fdrand_coo, solverbenchmark

function __init__()
    @require Pardiso="46dd5b70-b6fb-5a00-ae2d-e8fea33afaf2" include("factorizations/pardiso_lu.jl")
    @require IncompleteLU="40713840-3770-5561-ab4c-a76e7d0d7895" include("factorizations/ilut.jl")
    @require AlgebraicMultigrid="2169fc97-5a83-5252-b627-83903c6c433c" include("factorizations/amg.jl")
end

export ILUTPreconditioner, AMGPreconditioner
export PardisoLU, MKLPardisoLU, SparspakLU


end # module
