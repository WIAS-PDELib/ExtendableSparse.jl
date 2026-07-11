#
# This file defines `preconbuilders` which allow to specify `precs` preconditioner constructors
# for iterative solvers as they are handled in LinearSolve.jl.
# See https://docs.sciml.ai/LinearSolve/stable/basics/Preconditioners/#Specifying-Preconditioners
#

"""
        LinearSolvePreconBuilder(; method=UMFPACKFactorization())

Return callable object constructing a formal left preconditioner from a sparse LU factorization using LinearSolve
as the `precs` parameter for a  [`BlockPreconBuilder`](@ref)  or  iterative methods wrapped by LinearSolve.jl.
"""
Base.@kwdef struct LinearSolvePreconBuilder
    method = UMFPACKFactorization()
end
(::LinearSolvePreconBuilder)(A, p) = error("import LinearSolve in order to use LinearSolvePreconBuilder")


"""
    ILUZeroPreconBuilder(; blocksize = 1)

Return callable object constructing a left zero fill-in ILU preconditioner 
using [ILUZero.jl](https://github.com/mcovalt/ILUZero.jl)
"""
Base.@kwdef mutable struct ILUZeroPreconBuilder
    blocksize::Int = 1
end

function (b::ILUZeroPreconBuilder)(A0, p)
    A = SparseMatrixCSC(size(A0)..., getcolptr(A0), rowvals(A0), nonzeros(A0))
    return if b.blocksize == 1
        (ILUZero.ilu0(A), LinearAlgebra.I)
    else
        (ILU0BlockPrecon(ILUZero.ilu0(pointblock(A, b.blocksize), SVector{b.blocksize, eltype(A)})), LinearAlgebra.I)
    end
end

"""
    ILUTPreconBuilder(; droptol = 0.1)

Return callable object constructing a left ILUT preconditioner 
using [IncompleteLU.jl](https://github.com/haampie/IncompleteLU.jl)
"""
Base.@kwdef struct ILUTPreconBuilder
    droptol::Float64 = 0.1
end
(::ILUTPreconBuilder)(A, p) = error("import IncompleteLU.jl in order to use ILUTBuilder")


"""
     BlockPreconBuilder(;precs=UMFPACKPreconBuilder(),  
                         partitioning = A -> [1:size(A,1)]

Return callable object constructing a left block Jacobi preconditioner 
from partition of unknowns.

- `partitioning(A)`  shall return a vector of AbstractVectors describing the indices of the partitions of the matrix. 
  For a matrix of size `n x n`, e.g. partitioning could be `[ 1:n÷2, (n÷2+1):n]` or [ 1:2:n, 2:2:n].

- `precs(A,p)` shall return a left precondioner for a matrix block. It may be one function used for each partition
   or a vector of functions - one for each partition.
"""
Base.@kwdef mutable struct BlockPreconBuilder
    precs = UMFPACKPreconBuilder()
    partitioning = A -> [1:size(A, 1)]
end

function (blockprecs::BlockPreconBuilder)(A, p)
    (; precs, partitioning) = blockprecs
    Apart = partitioning(A)
    npart = length(Apart)
    if isa(precs, Vector)
        factorizations = [A -> precs[i](A, p)[1] for i in 1:npart]
    else
        factorizations = [A -> precs(A, p)[1] for i in 1:npart]
    end
    bp = BlockPreconditioner(A; partitioning = Apart, factorizations)
    return (bp, LinearAlgebra.I)
end


"""
    JacobiPreconBuilder(; blocksize = 1)

Return callable object constructing a left Jacobi preconditioner
to be passed as the `precs` parameter to iterative methods wrapped by LinearSolve.jl.
"""
Base.@kwdef struct JacobiPreconBuilder
    blocksize::Int = 1
end

(b::JacobiPreconBuilder)(A::AbstractSparseMatrixCSC, p) = (JacobiPreconditioner(A; blocksize = b.blocksize), LinearAlgebra.I)


"""
    ProductPreconBuilder(precs1, precs2)

Return LinearSolve `precs` compatible callable object which constructs
a  [`ProductPreconditioner`](@ref) from the  `precs1` and `precs2` preconbuilders.
"""
Base.@kwdef mutable struct ProductPreconBuilder
    precs1 = JacobiPreconBuilder()
    precs2 = JacobiPreconBuilder()
end

function (prodprecs::ProductPreconBuilder)(A, p)
    M1 = prodprecs.precs1(A, p)[1]
    M2 = prodprecs.precs2(A, p)[1]
    return ProductPreconditioner(A, M1, M2), LinearAlgebra.I
end

"""
    struct IdentityPreconBuilder

LinearSolve `precs` compatible preonditioner constructor for trivial preonditioner.
"""
struct IdentityPreconBuilder
end

function (::IdentityPreconBuilder)(A, p)
    return LinearAlgebra.I, LinearAlgebra.I
end
