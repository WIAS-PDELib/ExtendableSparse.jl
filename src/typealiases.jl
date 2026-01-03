"""
    MTExtendableSparseMatrixCSC

Multithreaded extendable sparse matrix  (Experimental).

Aliased to [`GenericMTExtendableSparseMatrixCSC`](@ref) with [`SparseMatrixDILNKC`](@ref) scalar matrix parameter.
"""
const MTExtendableSparseMatrixCSC{Tv, Ti} = GenericMTExtendableSparseMatrixCSC{SparseMatrixDILNKC{Tv, Ti}, Tv, Ti}
MTExtendableSparseMatrixCSC(m, n, args...) = MTExtendableSparseMatrixCSC{Float64, Int64}(m, n, args...)

"""
    STExtendableSparseMatrixCSC

Single threaded extendable sparse matrix (Experimental).

Aliased to [`GenericExtendableSparseMatrixCSC`](@ref) with [`SparseMatrixLNK`](@ref) scalar matrix parameter.
"""
const STExtendableSparseMatrixCSC{Tv, Ti} = GenericExtendableSparseMatrixCSC{SparseMatrixLNK{Tv, Ti}, Tv, Ti}
STExtendableSparseMatrixCSC(::Type{Tv}, m::Number, n::Number) where {Tv} = STExtendableSparseMatrixCSC{Tv, Int64}(m, n)
STExtendableSparseMatrixCSC(m::Number, n::Number) = STExtendableSparseMatrixCSC(Float64, m, n)
function STExtendableSparseMatrixCSC(A::AbstractSparseMatrixCSC{Tv, Ti}) where {Tv, Ti <: Integer}
    return GenericExtendableSparseMatrixCSC(
        SparseMatrixCSC(A),
        SparseMatrixLNK{Tv, Ti}(size(A)...)
    )
end
STExtendableSparseMatrixCSC(D::Diagonal) = STExtendableSparseMatrixCSC(sparse(D))
STExtendableSparseMatrixCSC(I, J, V::AbstractVector) = STExtendableSparseMatrixCSC(sparse(I, J, V))
STExtendableSparseMatrixCSC(I, J, V::AbstractVector, m, n) = STExtendableSparseMatrixCSC(sparse(I, J, V, m, n))
STExtendableSparseMatrixCSC(I, J, V::AbstractVector, combine::Function) = STExtendableSparseMatrixCSC(sparse(I, J, V, combine))
STExtendableSparseMatrixCSC(I, J, V::AbstractVector, m, n, combine::Function) = STExtendableSparseMatrixCSC(sparse(I, J, V, m, n, combine))


"""
    ExtendableSparseMatrixCSC

Aliased to  [`STExtendableSparseMatrixCSC`](@ref) to ensure backward compatibility
to ExtendableSparse v1.x.
"""
const ExtendableSparseMatrixCSC = STExtendableSparseMatrixCSC


"""
    ExtendableSparseMatrix

Aliased to  [`STExtendableSparseMatrixCSC`](@ref) to ensure backward compatibility
to ExtendableSparse v1.x.
"""
const ExtendableSparseMatrix = STExtendableSparseMatrixCSC
