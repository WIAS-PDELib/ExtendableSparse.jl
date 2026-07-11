#
# This file defines a number of preconditioners uses by the preconbuilders.
#


"""
    allow_views(::preconditioner)

Factorizations on matrix partitions within a block preconditioner may or may not work with array views.
E.g. the umfpack factorization cannot work with views, while ILUZeroPreconditioner can.
Implementing a method for `allow_views` returning `false` resp. `true` allows to dispatch to the proper case.
"""
allow_views(::Any) = false
allow_views(::ILUZero.ILU0Precon) = true

#################################################################################
mutable struct JacobiPreconditioner{Tb, BSize}
    invdiag::Vector{Tb}
end

"""
    JacobiPreconditioner(A; blocksize=1)

Create a Jacobi preconditioner. If `blocksize>1` it is a point block preconditioner with blocks of
type of `StaticArrays.SMatrix`
"""
function JacobiPreconditioner(A::AbstractSparseMatrixCSC; blocksize = 1)
    n = size(A, 1)
    if blocksize == 1
        Tv = eltype(A)
        invdiag = Array{Tv, 1}(undef, n)
        @inbounds for i in 1:n
            invdiag[i] = one(Tv) / A[i, i]
        end
        return JacobiPreconditioner{Tv, blocksize}(invdiag)
    else
        n % blocksize == 0 || throw(ArgumentError("JacobiPreconditioner: size(A, 1)=$(n) must be divisible by blocksize=$(blocksize)"))
        Tb = SMatrix{blocksize, blocksize, eltype(A), blocksize^2}
        nblock = n ÷ blocksize
        invdiag = Vector{Tb}(undef, nblock)
        block = zeros(eltype(A), blocksize, blocksize)
        for iblock in 1:nblock
            @inbounds for i in 1:blocksize, j in 1:blocksize
                ii = (iblock - 1) * blocksize + i
                jj = (iblock - 1) * blocksize + j
                block[i, j] = A[ii, jj]
            end
            invdiag[iblock] = inv(SMatrix{blocksize, blocksize}(block))
        end
        return JacobiPreconditioner{Tb, blocksize}(invdiag)
    end
end

function LinearAlgebra.ldiv!(u::AbstractVector{Tu}, p::JacobiPreconditioner{Tb, BSize}, v::AbstractVector{Tv}) where {Tu, Tv, Tb, BSize}
    n = length(p.invdiag)
    if BSize == 1
        for i in 1:n
            @inbounds u[i] = p.invdiag[i] * v[i]
        end
    else
        bu = reinterpret(SVector{BSize, Tu}, u)
        bv = reinterpret(SVector{BSize, Tv}, v)
        for i in 1:n
            @inbounds bu[i] = p.invdiag[i] * bv[i]
        end
    end
    return u
end

LinearAlgebra.ldiv!(p::JacobiPreconditioner, v) = ldiv!(v, p, v)

allow_views(::JacobiPreconditioner{T, 1}) where {T} = true
allow_views(::JacobiPreconditioner{T, BSize}) where {T, BSize} = false

############################################################################
"""
        pointblock(A,blocksize)

Create a pointblock matrix with entries of type `StaticArrays.SMatrix` of size `blocksize x blocksize` from A.
"""
function pointblock(A0::ExtendableSparseMatrixCSC{Tv, Ti}, blocksize) where {Tv, Ti}
    A = SparseMatrixCSC(A0)
    colptr = A.colptr
    rowval = A.rowval
    nzval = A.nzval
    n = A.n
    block = zeros(Tv, blocksize, blocksize)
    nblock = n ÷ blocksize
    b = SMatrix{blocksize, blocksize, Tv, blocksize^2}(block)
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

"""
    ILU0BlockPrecon

Point-block preconditioner based on ILUZero.ilu0
"""
struct ILU0BlockPrecon{BSize, BSquare, Tv, Ti}
    ilu0::ILUZero.ILU0Precon{SMatrix{BSize, BSize, Tv, BSquare}, Ti, SVector{BSize, Tv}}
end

function LinearAlgebra.ldiv!(
        Y::Vector{Tv},
        A::ILU0BlockPrecon{BSize, BSquare, Tv, Ti},
        B::Vector{Tv}
    ) where {BSize, BSquare, Tv, Ti}
    BY = reinterpret(SVector{BSize, Tv}, Y)
    BB = reinterpret(SVector{BSize, Tv}, B)
    ldiv!(BY, A.ilu0, BB)
    return Y
end

#######################################################################################
struct BlockPreconditioner
    partitioning::Vector{AbstractVector}
    factorizations::Vector{Any}
end

"""
     BlockPreconditioner(A;partitioning, factorizations)
    
Create a block preconditioner from partition of unknowns given by `partitioning`, a vector of AbstractVectors describing the
indices of the partitions of the matrix. For a matrix of size `n x n`, e.g. partitioning could be `[ 1:n÷2, (n÷2+1):n]`
or [ 1:2:n, 2:2:n].

`factorizations` is a thread safe callable `factorizations(A)` (Function or struct) which allows to create a factorization (with `ldiv!` methods)
from a submatrix of A, or a vector thereof.
"""
function BlockPreconditioner(
        A::AbstractSparseMatrixCSC;
        partitioning = [1:size(A, 1)],
        factorizations = (A) -> nothing
    )
    nall = sum(length, partitioning)
    n = size(A, 1)
    if nall != n
        @error "BlockPreconditioner: sum(length,partitioning)=$(nall) but n=$(n)"
    end
    np = length(partitioning)

    if !isa(factorizations, Vector)
        factorizations = fill(factorizations, np)
    end

    facts = Vector{Any}(undef, np)
    Threads.@threads for ipart in 1:np
        AP = A[partitioning[ipart], partitioning[ipart]]
        facts[ipart] = factorizations[ipart](AP)
    end

    return BlockPreconditioner(partitioning, facts)
end

function LinearAlgebra.ldiv!(p::BlockPreconditioner, v)
    (; factorizations, partitioning) = p
    np = length(partitioning)

    Threads.@threads for ipart in 1:np
        if allow_views(factorizations[ipart])
            ldiv!(factorizations[ipart], view(v, partitioning[ipart]))
        else
            vv = v[partitioning[ipart]]
            ldiv!(factorizations[ipart], vv)
            view(v, partitioning[ipart]) .= vv
        end
    end
    return v
end

function LinearAlgebra.ldiv!(u, p::BlockPreconditioner, v)
    (; factorizations, partitioning) = p
    np = length(partitioning)
    Threads.@threads  for ipart in 1:np
        if allow_views(factorizations[ipart])
            ldiv!(view(u, partitioning[ipart]), factorizations[ipart], view(v, partitioning[ipart]))
        else
            uu = u[partitioning[ipart]]
            ldiv!(uu, factorizations[ipart], v[partitioning[ipart]])
            view(u, partitioning[ipart]) .= uu
        end
    end
    return u
end

Base.eltype(p::BlockPreconditioner) = eltype(p.factorizations[1])


#######################################################################################

"""
    ProductPreconditioner(A,M1,M2)

Product M of two left preconditioning steps using `M1` and `M2`, respectively.

The operation ``u=M^{-1}v`` is defined by two simple iteration steps
with two different preconditioners ``M_1`` and ``M_2``:

Let ``u_0=0``. Then calculate
```math    
 \\begin{align*}
   u_1&= u_0 - M_1^{-1}(Au_0 - v) = M_1^{-1}v\\\\
   u  &= u_1 - M_2^{-1}(Au_1 - v)
  \\end{align*}
```
"""
Base.@kwdef struct ProductPreconditioner{TA, TM1, TM2}
    A::TA
    M1::TM1
    M2::TM2
end

function LinearAlgebra.ldiv!(u, p::ProductPreconditioner, v)
    (; A, M1, M2) = p
    u1 = similar(u)
    u2 = similar(u)
    ldiv!(u1, M1, v)
    mul!(u2, A, u1)
    ldiv!(u, M2, v - u2)
    u .+= u1
    return u
end

function LinearAlgebra.ldiv!(p::ProductPreconditioner, v)
    u = ldiv!(copy(v), p, v)
    v .= u
    return v
end
