iluzerowarned = false
mutable struct ILUZeroPreconditioner <: AbstractPreconditioner
    A::ExtendableSparseMatrix
    factorization::ILUZero.ILU0Precon
    phash::UInt64
    function ILUZeroPreconditioner()
        global iluzerowarned
        if !iluzerowarned
            @warn "ILUZeroPreconditioner is deprecated. Use LinearSolve with `precs=ILUZeroPreconBuilder()` instead"
            iluzerowarned = true
        end
        p = new()
        p.phash = 0
        return p
    end
end

"""
```
ILUZeroPreconditioner()
ILUZeroPreconditioner(matrix)
```
Incomplete LU preconditioner with zero fill-in using  [ILUZero.jl](https://github.com/mcovalt/ILUZero.jl). This preconditioner
also calculates and stores updates to the off-diagonal entries and thus delivers better convergence than  the [`ILU0Preconditioner`](@ref).
"""
function ILUZeroPreconditioner end

function update!(p::ILUZeroPreconditioner)
    flush!(p.A)
    if p.A.phash != p.phash
        p.factorization = ILUZero.ilu0(p.A.cscmatrix)
        p.phash = p.A.phash
    else
        ILUZero.ilu0!(p.factorization, p.A.cscmatrix)
    end
    return p
end

allow_views(::ILUZeroPreconditioner) = true
allow_views(::Type{ILUZeroPreconditioner}) = true


biluzerowarned = false
mutable struct PointBlockILUZeroPreconditioner <: AbstractPreconditioner
    A::ExtendableSparseMatrix
    factorization::ILUZero.ILU0Precon
    phash::UInt64
    blocksize::Int
    function PointBlockILUZeroPreconditioner(; blocksize = 1)
        global biluzerowarned
        if !biluzerowarned
            @warn "PointBlockILUZeroPreconditioner is deprecated. Use LinearSolve with `precs=ILUZeroPreconBuilder(; blocksize=$(blocksize))` instead"
            biluzerowarned = true
        end
        p = new()
        p.phash = 0
        p.blocksize = blocksize
        return p
    end
end

"""
```
PointBlockILUZeroPreconditioner(;blocksize)
PointBlockILUZeroPreconditioner(matrix;blocksize)
```
Incomplete LU preconditioner with zero fill-in using  [ILUZero.jl](https://github.com/mcovalt/ILUZero.jl). This preconditioner
also calculates and stores updates to the off-diagonal entries and thus delivers better convergence than  the [`ILU0Preconditioner`](@ref).
"""
function PointBlockILUZeroPreconditioner end

function update!(p::PointBlockILUZeroPreconditioner)
    flush!(p.A)
    Ab = pointblock(p.A.cscmatrix, p.blocksize)
    if p.A.phash != p.phash
        p.factorization = ILUZero.ilu0(Ab, SVector{p.blocksize, eltype(p.A)})
        p.phash = p.A.phash
    else
        ILUZero.ilu0!(p.factorization, Ab)
    end
    return p
end


function LinearAlgebra.ldiv!(p::PointBlockILUZeroPreconditioner, v)
    vv = reinterpret(SVector{p.blocksize, eltype(v)}, v)
    LinearAlgebra.ldiv!(vv, p.factorization, vv)
    return v
end

function LinearAlgebra.ldiv!(u, p::PointBlockILUZeroPreconditioner, v)
    LinearAlgebra.ldiv!(
        reinterpret(SVector{p.blocksize, eltype(u)}, u),
        p.factorization,
        reinterpret(SVector{p.blocksize, eltype(v)}, v)
    )
    return u
end
