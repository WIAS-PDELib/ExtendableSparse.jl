
struct UMFPACKPreconBuilder end
(::UMFPACKPreconBuilder)(A::AbstractSparseMatrixCSC,p)=(SparseArrays.UMFPACK.UmfpackLU(SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A),nonzeros(A))),LinearAlgebra.I)

struct SparspakPreconBuilder end
(::SparspakPreconBuilder)(A::AbstractSparseMatrixCSC,p)=(sparspaklu(SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A),nonzeros(A))),LinearAlgebra.I)

struct JacobiPreconBuilder end
(::JacobiPreconBuilder)(A::AbstractSparseMatrixCSC,p)=(JacobiPreconditioner(SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A),nonzeros(A))),LinearAlgebra.I)


struct ILUZeroBuilder end
(::ILUZeroBuilder)(A,p)=(ilu0(A),LinearAlgebra.I)



Base.@kwdef struct ILUTBuilder
    droptol::Float64=0.1
end
(::ILUTBuilder)(A,p)= error("import IncompleteLU.jl in order to use ILUTBuilder")

struct SmoothedAggregationAMGBuilder{Tk}
    blocksize::Int
    kwargs::Tk
end

function SmoothedAggregationAMGBuilder(;blocksize=1, kwargs...)
    SmoothedAggregationAMGBuilder(blocksize,kwargs)
end

(::SmoothedAggregationAMGBuilder)(A,p)= error("import AlgebraicMultigrid in order to use SmoothedAggregationAMGBuilder")

struct RugeStubenAMGBuilder{Tk}
    blocksize::Int
    kwargs::Tk
end

function RugeStubenAMGBuilder(;blocksize=1, kwargs...)
    SmoothedAggregationAMGBuilder(blocksize,kwargs)
end

(::RugeStubenAMGBuilder)(A,p)= error("import AlgebraicMultigrid in order to use RugeStubenAMGBuilder")
