module ExtendableSparseAlgebraicMultigridExt
using ExtendableSparse
using AlgebraicMultigrid: AlgebraicMultigrid, ruge_stuben, smoothed_aggregation, aspreconditioner
using SparseArrays: SparseMatrixCSC, AbstractSparseMatrixCSC
using LinearAlgebra: I


import ExtendableSparse: SmoothedAggregationAMGBuilder
import ExtendableSparse: RugeStubenAMGBuilder

(b::SmoothedAggregationAMGBuilder)(A::AbstractSparseMatrixCSC,p)= (aspreconditioner(smoothed_aggregation(SparseMatrixCSC(A), Val{b.blocksize}; b.kwargs...)),I)
(b::RugeStubenAMGBuilder)(A::AbstractSparseMatrixCSC,p)= (aspreconditioner(ruge_stuben(SparseMatrixCSC(A), Val{b.blocksize}; b.kwargs...)),I)


####
# Deprecated from here on
# TODO remove in v2.0

import ExtendableSparse: @makefrommatrix, AbstractPreconditioner, update!

######################################################################################
mutable struct RS_AMGPreconditioner <: AbstractPreconditioner
    A::ExtendableSparseMatrix
    factorization::AlgebraicMultigrid.Preconditioner
    kwargs
    blocksize
    function ExtendableSparse.RS_AMGPreconditioner(blocksize=1; kwargs...)
        precon = new()
        precon.kwargs = kwargs
        precon.blocksize=blocksize
        precon
    end
end


@eval begin
    @makefrommatrix  ExtendableSparse.RS_AMGPreconditioner
end

function update!(precon::RS_AMGPreconditioner)
    @inbounds flush!(precon.A)
    precon.factorization =  AlgebraicMultigrid.aspreconditioner(AlgebraicMultigrid.ruge_stuben(precon.A.cscmatrix,Val{precon.blocksize}; precon.kwargs...))
end

allow_views(::RS_AMGPreconditioner)=true
allow_views(::Type{RS_AMGPreconditioner})=true

######################################################################################
mutable struct SA_AMGPreconditioner <: AbstractPreconditioner
    A::ExtendableSparseMatrix
    factorization::AlgebraicMultigrid.Preconditioner
    kwargs
    blocksize
    function ExtendableSparse.SA_AMGPreconditioner(blocksize=1; kwargs...)
        precon = new()
        precon.kwargs = kwargs
        precon.blocksize=blocksize
        precon
    end
end


@eval begin
    @makefrommatrix  ExtendableSparse.SA_AMGPreconditioner
end

function update!(precon::SA_AMGPreconditioner)
    @inbounds flush!(precon.A)
    precon.factorization =  AlgebraicMultigrid.aspreconditioner(AlgebraicMultigrid.smoothed_aggregation(precon.A.cscmatrix, Val{precon.blocksize}; precon.kwargs...))
end

allow_views(::SA_AMGPreconditioner)=true
allow_views(::Type{SA_AMGPreconditioner})=true

######################################################################################
# deprecated
# mutable struct AMGPreconditioner <: AbstractPreconditioner
#     A::ExtendableSparseMatrix
#     factorization::AlgebraicMultigrid.Preconditioner
#     max_levels::Int
#     max_coarse::Int
#     function ExtendableSparse.AMGPreconditioner(; max_levels = 10, max_coarse = 10)
#         precon = new()
#         precon.max_levels = max_levels
#         precon.max_coarse = max_coarse
#         precon
#     end
# end


# @eval begin
#     @makefrommatrix  ExtendableSparse.AMGPreconditioner
# end

# function update!(precon::AMGPreconditioner)
#     @inbounds flush!(precon.A)
#     precon.factorization = AlgebraicMultigrid.aspreconditioner(AlgebraicMultigrid.ruge_stuben(precon.A.cscmatrix))
# end

# allow_views(::AMGPreconditioner)=true
# allow_views(::Type{AMGPreconditioner})=true

end
