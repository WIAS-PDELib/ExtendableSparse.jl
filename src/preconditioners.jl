"""
    allow_views(::preconditioner)

Factorizations on matrix partitions within a block preconditioner may or may not work with array views.
E.g. the umfpack factorization cannot work with views, while ILUZeroPreconditioner can.
 Implementing a method for `allow_views` returning `false` resp. `true` allows to dispatch to the proper case.
"""
allow_views(::Any) = false

mutable struct JacobiPreconditioner{Tb, N}
    invdiag::Vector{Tb}
end


"""
    JacobiPreconditioner(A; blocksize=1)
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
        Tb = SMatrix{blocksize, blocksize, eltype(A), blocksize^2}
        nblock = n ÷ blocksize
        invdiag = Array{Tb, 1}(undef, nblock)
        block = zeros(eltype(A), blocksize, blocksize)
        for iblock in 1:nblock
            for i in 1:blocksize
                for j in 1:blocksize
                    ii = (iblock - 1) * blocksize + i
                    jj = (iblock - 1) * blocksize + j
                    block[i, j] = A[ii, jj]
                    sblock = SMatrix{blocksize, blocksize}(block)
                    invdiag[iblock] = inv(sblock)
                end
            end
        end
        return JacobiPreconditioner{Tb, blocksize}(invdiag)
    end
end


function LinearAlgebra.ldiv!(u::AbstractVector{Tu}, p::JacobiPreconditioner{Tb, N}, v::AbstractVector{Tv}) where {Tu, Tv, Tb, N}
    n = length(p.invdiag)
    if N == 1
        for i in 1:n
            @inbounds u[i] = p.invdiag[i] * v[i]
        end
    else
        bu = reinterpret(SVector{N, Tu}, u)
        bv = reinterpret(SVector{N, Tv}, v)
        for i in 1:n
            @inbounds bu[i] = p.invdiag[i] * bv[i]
        end
    end
    return u
end

LinearAlgebra.ldiv!(p::JacobiPreconditioner, v) = ldiv!(v, p, v)

allow_views(::JacobiPreconditioner{T, 1}) where {T} = true
allow_views(::JacobiPreconditioner{T, N}) where {T, N} = true


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

struct ILU0BlockPrecon{N, NN, Tv, Ti}
    ilu0::ILUZero.ILU0Precon{SMatrix{N, N, Tv, NN}, Ti, SVector{N, Tv}}
end

function LinearAlgebra.ldiv!(
        Y::Vector{Tv},
        A::ILU0BlockPrecon{N, NN, Tv, Ti},
        B::Vector{Tv}
    ) where {N, NN, Tv, Ti}
    BY = reinterpret(SVector{N, Tv}, Y)
    BB = reinterpret(SVector{N, Tv}, B)
    ldiv!(BY, A.ilu0, BB)
    return Y
end

######
mutable struct BlockPreconditioner
    A::AbstractMatrix
    factorizations
    partitioning::Union{Nothing, Vector{AbstractVector}}
    facts::Vector
    function BlockPreconditioner(A; partitioning = nothing, factorization = nothing, factorizations = nothing)
        p = new()
        p.A = A
        p.partitioning = partitioning
        if !isnothing(factorization)
            p.factorizations = [factorization for i in 1:length(partitioning)]
        else
            p.factorizations = factorizations
        end
        update!(p)
        return p
    end
end


"""
     BlockPreconditioner(;partitioning, factorization)
    
Create a block preconditioner from partition of unknowns given by `partitioning`, a vector of AbstractVectors describing the
indices of the partitions of the matrix. For a matrix of size `n x n`, e.g. partitioning could be `[ 1:n÷2, (n÷2+1):n]`
or [ 1:2:n, 2:2:n].
Factorization is a callable (Function or struct) which allows to create a factorization (with `ldiv!` methods) from a submatrix of A.
"""
function BlockPreconditioner end


function update!(precon::BlockPreconditioner)
    flush!(precon.A)
    nall = sum(length, precon.partitioning)
    n = size(precon.A, 1)
    if nall != n
        @warn "sum(length,partitioning)=$(nall) but n=$(n)"
    end

    if isnothing(precon.partitioning)
        partitioning = [1:n]
    end

    np = length(precon.partitioning)
    precon.facts = Vector{Any}(undef, np)
    Threads.@threads for ipart in 1:np
        AP = precon.A[precon.partitioning[ipart], precon.partitioning[ipart]]
        precon.facts[ipart] = precon.factorizations[ipart](AP)
    end
    return
end


function LinearAlgebra.ldiv!(p::BlockPreconditioner, v)
    partitioning = p.partitioning
    facts = p.facts
    np = length(partitioning)

    Threads.@threads for ipart in 1:np
        if allow_views(p.factorizations[ipart])
            ldiv!(facts[ipart], view(v, partitioning[ipart]))
        else
            vv = v[partitioning[ipart]]
            ldiv!(facts[ipart], vv)
            view(v, partitioning[ipart]) .= vv
        end
    end
    return v
end

function LinearAlgebra.ldiv!(u, p::BlockPreconditioner, v)
    partitioning = p.partitioning
    facts = p.facts
    np = length(partitioning)
    Threads.@threads  for ipart in 1:np
        if allow_views(p.facts[ipart])
            ldiv!(view(u, partitioning[ipart]), facts[ipart], view(v, partitioning[ipart]))
        else
            uu = u[partitioning[ipart]]
            ldiv!(uu, facts[ipart], v[partitioning[ipart]])
            view(u, partitioning[ipart]) .= uu
        end
    end
    return u
end

Base.eltype(p::BlockPreconditioner) = eltype(p.facts[1])


"""
    struct ProductPreconditioner

Product of two left preconditioning steps.
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
