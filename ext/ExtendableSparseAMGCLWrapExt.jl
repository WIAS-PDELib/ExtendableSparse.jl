# The whole extension is deprecated
# TODO remove in v2.0
module ExtendableSparseAMGCLWrapExt
using ExtendableSparse
using AMGCLWrap

import ExtendableSparse: @makefrommatrix, AbstractPreconditioner, update!

#############################################################################
mutable struct AMGCL_AMGPreconditioner <: AbstractPreconditioner
    A::ExtendableSparseMatrix
    factorization::AMGCLWrap.AMGPrecon
    kwargs
    function ExtendableSparse.AMGCL_AMGPreconditioner(; kwargs...)
        Base.depwarn("AMGCL_AMGPreconditioner() is deprecated. Use LinearSolve with `precs=AMGCLWrap.AMGPreconBuilder()`  instead.",
                     :AMGCL_AMGPreconditioner)
        precon = new()
        precon.kwargs = kwargs
        precon
    end
end


@eval begin
    @makefrommatrix  ExtendableSparse.AMGCL_AMGPreconditioner
end

function update!(precon::AMGCL_AMGPreconditioner)
    @inbounds flush!(precon.A)
    precon.factorization = AMGCLWrap.AMGPrecon(precon.A;precon.kwargs...)
end

allow_views(::AMGCL_AMGPreconditioner)=true
allow_views(::Type{AMGCL_AMGPreconditioner})=true

#############################################################################
mutable struct AMGCL_RLXPreconditioner <: AbstractPreconditioner
    A::ExtendableSparseMatrix
    factorization::AMGCLWrap.RLXPrecon
    kwargs
    function ExtendableSparse.AMGCL_RLXPreconditioner(; kwargs...)
        Base.depwarn("AMGCL_RLXPreconditioner() is deprecated. Use LinearSolve with  `precs=AMGCLWrap.RLXPreconBuilder()`  instead.",
                     :AMGCL_RLXPreconditioner)
        precon = new()
        precon.kwargs = kwargs
        precon
    end
end


@eval begin
    @makefrommatrix  ExtendableSparse.AMGCL_RLXPreconditioner
end

function update!(precon::AMGCL_RLXPreconditioner)
    @inbounds flush!(precon.A)
    precon.factorization = AMGCLWrap.RLXPrecon(precon.A;precon.kwargs...)
end

allow_views(::AMGCL_RLXPreconditioner)=true
allow_views(::Type{AMGCL_RLXPreconditioner})=true



end
