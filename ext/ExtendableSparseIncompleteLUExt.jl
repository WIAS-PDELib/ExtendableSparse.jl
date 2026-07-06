module ExtendableSparseIncompleteLUExt
using ExtendableSparse
using IncompleteLU: IncompleteLU
using LinearAlgebra: I
using SparseArrays: AbstractSparseMatrixCSC, SparseMatrixCSC

import ExtendableSparse: ILUTPreconBuilder

(b::ILUTPreconBuilder)(A::AbstractSparseMatrixCSC, p) = (IncompleteLU.ilu(SparseMatrixCSC(A); τ = b.droptol), I)


end
