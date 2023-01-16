abstract type AbstractPardisoLU{Tv, Ti} <: AbstractLUFactorization{Tv, Ti} end

mutable struct PardisoLU{Tv, Ti} <: AbstractPardisoLU{Tv, Ti}
    A::Union{ExtendableSparseMatrix{Tv, Ti}, Nothing}
    ps::Pardiso.PardisoSolver
    phash::UInt64
end

function PardisoLU{Tv, Ti}(; iparm = nothing, dparm = nothing,
                           mtype = nothing) where {Tv, Ti}
    fact = PardisoLU{Tv, Ti}(nothing, Pardiso.PardisoSolver(), 0)
    default_initialize!(fact, iparm, dparm, mtype)
end

"""
```
PardisoLU(;valuetype=Float64, 
           indextype=Int64,
           iparm::Vector, 
           dparm::Vector, 
           mtype::Int)

PardisoLU(matrix; iparm,dparm,mtype)
```

LU factorization based on pardiso. For using this, you need to issue `using Pardiso`
and have the pardiso library from  [pardiso-project.org](https://pardiso-project.org) 
[installed](https://github.com/JuliaSparse/Pardiso.jl#pardiso-60).

The optional keyword arguments `mtype`, `iparm`  and `dparm` are 
(Pardiso internal parameters)[https://github.com/JuliaSparse/Pardiso.jl#readme].

Forsetting them, one can also access the `PardisoSolver` e.g. like
```
using Pardiso
plu=PardisoLU()
Pardiso.set_iparm!(plu.ps,5,13.0)
```
"""
function PardisoLU(; valuetype::Type = Float64, indextype::Type = Int64, kwargs...)
    PardisoLU{valuetype, indextype}(; kwargs...)
end

#############################################################################################
mutable struct MKLPardisoLU{Tv, Ti} <: AbstractPardisoLU{Tv, Ti}
    A::Union{ExtendableSparseMatrix{Tv, Ti}, Nothing}
    ps::Pardiso.MKLPardisoSolver
    phash::UInt64
end

function MKLPardisoLU{Tv, Ti}(; iparm = nothing, mtype = nothing) where {Tv, Ti}
    fact = MKLPardisoLU{Tv, Ti}(nothing, Pardiso.MKLPardisoSolver(), 0)
    default_initialize!(fact, iparm, nothing, mtype)
end

"""
```
MKLPardisoLU(;valuetype=Float64, 
           indextype=Int64,
           iparm::Vector, 
           mtype::Int)

MKLPardisoLU(matrix; iparm, mtype)
```

LU factorization based on pardiso. For using this, you need to issue `using Pardiso`.
This version  uses the early 2000's fork in Intel's MKL library.

The optional keyword arguments `mtype` and `iparm`  are  
(Pardiso internal parameters)[https://github.com/JuliaSparse/Pardiso.jl#readme].

For setting them you can also access the `PardisoSolver` e.g. like
```
using Pardiso
plu=MKLPardisoLU()
Pardiso.set_iparm!(plu.ps,5,13.0)
```
"""
function MKLPardisoLU(; valuetype::Type = Float64, indextype::Type = Int64, kwargs...)
    MKLPardisoLU{valuetype, indextype}(; kwargs...)
end

##########################################################################################
function default_initialize!(fact::AbstractPardisoLU{Tv, Ti},
                             iparm,
                             dparm,
                             mtype) where {Tv, Ti}
    # if !isnothing(mtype)
    #     my_mtype=mtype fix this!
    # else

    if Tv <: Complex
        my_mtype = Pardiso.COMPLEX_NONSYM
    else
        my_mtype = Pardiso.REAL_NONSYM
    end

    Pardiso.set_matrixtype!(fact.ps, my_mtype)

    if !isnothing(iparm)
        for i = 1:min(length(iparm), length(fact.ps.iparm))
            Pardiso.set_iparm!(fact.ps, i, iparm[i])
        end
    end

    if !isnothing(dparm)
        for i = 1:min(length(dparm), length(fact.ps.dparm))
            Pardiso.set_dparm!(fact.ps, i, dparm[i])
        end
    end
    fact
end

function update!(lufact::AbstractPardisoLU{Tv, Ti}) where {Tv, Ti}
    ps = lufact.ps
    flush!(lufact.A)
    Acsc = lufact.A.cscmatrix
    if lufact.phash != lufact.A.phash
        Pardiso.set_phase!(ps, Pardiso.RELEASE_ALL)
        Pardiso.pardiso(ps, Tv[], Acsc, Tv[])
        Pardiso.set_phase!(ps, Pardiso.ANALYSIS_NUM_FACT)
        lufact.phash = lufact.A.phash
    else
        Pardiso.set_phase!(ps, Pardiso.NUM_FACT)
    end
    Pardiso.fix_iparm!(ps, :N)
    Pardiso.pardiso(ps, Tv[], Acsc, Tv[])
    lufact
end

function LinearAlgebra.ldiv!(u::AbstractArray{T, 1} where {T},
                             lufact::AbstractPardisoLU,
                             v::AbstractArray{T, 1} where {T})
    ps = lufact.ps
    Acsc = lufact.A.cscmatrix
    Pardiso.set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    Pardiso.pardiso(ps, u, Acsc, v)
    u
end

function LinearAlgebra.ldiv!(fact::AbstractPardisoLU, v::AbstractArray{T, 1} where {T})
    ldiv!(v, fact, copy(v))
end

@eval begin
    @makefrommatrix PardisoLU
    @makefrommatrix MKLPardisoLU
end
