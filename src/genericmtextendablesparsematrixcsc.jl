"""
    $(TYPEDEF)


Extendable sparse matrix parametrized by sparse matrix extension allowing multithreaded assembly and
parallel matrix-vector multiplication.

Fields:
- `cscmatrix`: a SparseMatrixCSC  containing existing matrix entries
- `xmatrices`: vector of instances of [`AbstractSparseMatrixExtension`](@ref) used to collect new entries
- `colparts`: vector describing colors of the partitions of the unknowns
- `partnodes`: vector describing partition of the unknowns

It is assumed that the set of unknowns is partitioned, and the partitioning is colored in such a way that
several partitions of the same color can be handled by different threads, both during matrix assembly (which
in general would use a partition of e.g. finite elements compatible to the partitioning of the nodes) and during
matrix-vector multiplication. This approach is compatible with the current choice of the standard Julia
sparse  ecosystem which prefers compressed colume storage (CSC) over compressed row storage (CSR).

"""
mutable struct GenericMTExtendableSparseMatrixCSC{Tm <: AbstractSparseMatrixExtension, Tv, Ti <: Integer} <: AbstractExtendableSparseMatrixCSC{Tv, Ti}
    cscmatrix::SparseMatrixCSC{Tv, Ti}
    xmatrices::Vector{Tm}
    colparts::Vector{Ti}
    partnodes::Vector{Ti}
end

function GenericMTExtendableSparseMatrixCSC{Tm, Tv, Ti}(n, m, p::Integer = 1) where {Tm <: AbstractSparseMatrixExtension, Tv, Ti}
    return GenericMTExtendableSparseMatrixCSC(
        spzeros(Tv, Ti, m, n),
        [Tm(m, n) for i in 1:p],
        Ti[1, 2],
        Ti[1, n + 1],
    )
end

"""
    $(TYPEDSIGNATURES)

Set node partitioning.
"""
function partitioning!(ext::GenericMTExtendableSparseMatrixCSC{Tm, Tv, Ti}, colparts, partnodes) where {Tm, Tv, Ti}
    ext.partnodes = partnodes
    ext.colparts = colparts
    return ext
end


"""
    $(TYPEDSIGNATURES)
"""
function reset!(ext::GenericMTExtendableSparseMatrixCSC{Tm, Tv, Ti}, p::Integer) where {Tm, Tv, Ti}
    m, n = size(ext.cscmatrix)
    ext.cscmatrix = spzeros(Tv, Ti, m, n)
    ext.xmatrices = [Tm(m, n) for i in 1:p]
    ext.colparts = Ti[1, 2]
    ext.partnodes = Ti[1, n + 1]
    return ext
end

"""
    $(TYPEDSIGNATURES)
"""
function reset!(ext::GenericMTExtendableSparseMatrixCSC)
    return reset!(ext, length(ext.xmatrices))
end


"""
    $(TYPEDSIGNATURES)
"""
function flush!(ext::GenericMTExtendableSparseMatrixCSC{Tm, Tv, Ti}) where {Tm, Tv, Ti}
    ext.cscmatrix = Base.sum(ext.xmatrices, ext.cscmatrix)
    np = length(ext.xmatrices)
    (m, n) = size(ext.cscmatrix)
    ext.xmatrices = [Tm(m, n) for i in 1:np]
    return ext
end


"""
    $(TYPEDSIGNATURES)
"""
function SparseArrays.sparse(ext::GenericMTExtendableSparseMatrixCSC)
    flush!(ext)
    return ext.cscmatrix
end

"""
    $(TYPEDSIGNATURES)
"""
function Base.setindex!(
        ext::GenericMTExtendableSparseMatrixCSC,
        v::Any,
        i::Integer,
        j::Integer
    )
    k = findindex(ext.cscmatrix, i, j)
    return if k > 0
        ext.cscmatrix.nzval[k] = v
    else
        error("use rawupdateindex! for new entries into GenericMTExtendableSparseMatrixCSC")
    end
end

# to resolve ambiguity
function Base.setindex!(
        ext::GenericMTExtendableSparseMatrixCSC,
        v::AbstractVecOrMat,
        i::Integer,
        j::Integer
    )
    k = findindex(ext.cscmatrix, i, j)
    return if k > 0
        ext.cscmatrix.nzval[k] = v
    else
        error("use rawupdateindex! for new entries into GenericMTExtendableSparseMatrixCSC")
    end
end


"""
    $(TYPEDSIGNATURES)
"""
function Base.getindex(
        ext::GenericMTExtendableSparseMatrixCSC,
        i::Integer,
        j::Integer
    )
    k = findindex(ext.cscmatrix, i, j)
    if k > 0
        return ext.cscmatrix.nzval[k]
    elseif sum(nnz, ext.xmatrices) == 0
        return zero(eltype(ext.cscmatrix))
    else
        error("flush! GenericMTExtendableSparseMatrixCSC before using getindex")
    end
end

"""
    $(TYPEDSIGNATURES)
"""
nnznew(ext::GenericMTExtendableSparseMatrixCSC) = sum(nnz, ext.xmatrices)


"""
    $(TYPEDSIGNATURES)
"""
function rawupdateindex!(
        ext::GenericMTExtendableSparseMatrixCSC,
        op,
        v,
        i,
        j,
        tid = 1
    )
    k = findindex(ext.cscmatrix, i, j)
    return if k > 0
        ext.cscmatrix.nzval[k] = op(ext.cscmatrix.nzval[k], v)
    else
        rawupdateindex!(ext.xmatrices[tid], op, v, i, j)
    end
end


"""
    $(TYPEDSIGNATURES)
"""
function updateindex!(
        ext::GenericMTExtendableSparseMatrixCSC,
        op,
        v,
        i,
        j,
        tid = 1
    )
    k = findindex(ext.cscmatrix, i, j)
    return if k > 0
        ext.cscmatrix.nzval[k] = op(ext.cscmatrix.nzval[k], v)
    else
        updateindex!(ext.xmatrices[tid], op, v, i, j)
    end
end


# Needed in 1.9
function Base.:*(ext::GenericMTExtendableSparseMatrixCSC{Tm, TA} where {Tm <: ExtendableSparse.AbstractSparseMatrixExtension}, x::Union{StridedVector, BitVector}) where {TA}
    return mul!(similar(x), ext, x)
end

"""
    $(TYPEDSIGNATURES)
"""
function LinearAlgebra.mul!(r::AbstractVecOrMat, ext::GenericMTExtendableSparseMatrixCSC, x::AbstractVecOrMat)
    flush!(ext)
    A = ext.cscmatrix
    colparts = ext.colparts
    partnodes = ext.partnodes
    rows = SparseArrays.rowvals(A)
    vals = nonzeros(A)
    r .= zero(eltype(ext))
    m, n = size(A)
    for icol in 1:(length(colparts) - 1)
        Threads.@threads for ip in colparts[icol]:(colparts[icol + 1] - 1)
            @inbounds for inode in partnodes[ip]:(partnodes[ip + 1] - 1)
                @inbounds for i in nzrange(A, inode)
                    r[rows[i]] += vals[i] * x[inode]
                end
            end
        end
    end
    return r
end

# to resolve ambiguity
function LinearAlgebra.mul!(::SparseArrays.AbstractSparseMatrixCSC, ::ExtendableSparse.GenericMTExtendableSparseMatrixCSC, ::LinearAlgebra.Diagonal)
    throw(MethodError("mul!(::AbstractSparseMatrixCSC, ::GenericMTExtendableSparseMatrixCSC,::Diagonal) is impossible"))
    return nothing
end

function LinearAlgebra.mul!(::AbstractMatrix, ::ExtendableSparse.GenericMTExtendableSparseMatrixCSC, ::LinearAlgebra.AbstractTriangular)
    throw(MethodError("mul!(::AbstractMatrix, ::GenericMTExtendableSparseMatrixCSC, ::AbstractTriangular) is impossible"))
    return nothing
end
