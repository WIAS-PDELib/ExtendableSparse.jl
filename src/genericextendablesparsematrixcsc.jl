"""
    $(TYPEDEF)

Single threaded extendable sparse matrix parametrized by sparse matrix extension.
"""
mutable struct GenericExtendableSparseMatrixCSC{Tm <: AbstractSparseMatrixExtension, Tv, Ti <: Integer} <: AbstractExtendableSparseMatrixCSC{Tv, Ti}
    """
    Final matrix data
    """
    cscmatrix::SparseMatrixCSC{Tv, Ti}

    """
    Matrix for new entries
    """
    xmatrix::Tm
end


function GenericExtendableSparseMatrixCSC{Tm, Tv, Ti}(m::Integer, n::Integer) where {Tm <: AbstractSparseMatrixExtension, Tv, Ti <: Integer}
    return GenericExtendableSparseMatrixCSC(
        spzeros(Tv, Ti, m, n),
        Tm(m, n)
    )
end

function GenericExtendableSparseMatrixCSC{Tm, Tv, Ti}(A::SparseMatrixCSC{Tv, Ti}) where {Tm <: AbstractSparseMatrixExtension, Tv, Ti <: Integer}
    return GenericExtendableSparseMatrixCSC(
        SparseMatrixCSC(A),
        Tm(size(A)...)
    )
end

"""
    $(TYPEDSIGNATURES)
"""
function Base.similar(m::GenericExtendableSparseMatrixCSC{Tm, Tv, Ti}) where {Tm, Tv, Ti}
    return ExtendableSparseMatrixCSC{Tv, Ti}(size(m)...)
end

"""
    $(TYPEDSIGNATURES)
"""
function Base.similar(m::GenericExtendableSparseMatrixCSC{Tm, Tv, Ti}, ::Type{T}) where {Tm, Tv, Ti, T}
    return ExtendableSparseMatrixCSC{T, Ti}(size(m)...)
end


"""
    $(TYPEDSIGNATURES)
"""
nnznew(ext::GenericExtendableSparseMatrixCSC) = nnz(ext.xmatrix)

"""
    $(TYPEDSIGNATURES)
"""
function reset!(ext::GenericExtendableSparseMatrixCSC{Tm, Tv, Ti}) where {Tm, Tv, Ti}
    m, n = size(ext.cscmatrix)
    ext.cscmatrix = spzeros(Tv, Ti, m, n)
    ext.xmatrix = Tm(m, n)
    return ext
end


"""
    $(TYPEDSIGNATURES)
"""
function flush!(ext::GenericExtendableSparseMatrixCSC{Tm, Tv, Ti}) where {Tm, Tv, Ti}
    if nnz(ext.xmatrix) > 0
        ext.cscmatrix = ext.xmatrix + ext.cscmatrix
        ext.xmatrix = Tm(size(ext.cscmatrix)...)
    end
    return ext
end

"""
    $(TYPEDSIGNATURES)
"""
function SparseArrays.sparse(ext::GenericExtendableSparseMatrixCSC)
    flush!(ext)
    return ext.cscmatrix
end

"""
    $(TYPEDSIGNATURES)
"""
function Base.setindex!(
        ext::GenericExtendableSparseMatrixCSC,
        v::Any,
        i::Integer,
        j::Integer
    )
    k = findindex(ext.cscmatrix, i, j)
    return if k > 0
        ext.cscmatrix.nzval[k] = v
    else
        setindex!(ext.xmatrix, v, i, j)
    end
end

# to resolve ambiguity
function Base.setindex!(
        ext::GenericExtendableSparseMatrixCSC,
        v::AbstractVecOrMat,
        i::Integer,
        j::Integer
    )
    k = findindex(ext.cscmatrix, i, j)
    return if k > 0
        ext.cscmatrix.nzval[k] = v
    else
        setindex!(ext.xmatrix, v, i, j)
    end
end


"""
    $(TYPEDSIGNATURES)
"""
function Base.getindex(
        ext::GenericExtendableSparseMatrixCSC,
        i::Integer,
        j::Integer
    )
    k = findindex(ext.cscmatrix, i, j)
    return if k > 0
        ext.cscmatrix.nzval[k]
    else
        getindex(ext.xmatrix, i, j)
    end
end

"""
    $(TYPEDSIGNATURES)
"""
function rawupdateindex!(
        ext::GenericExtendableSparseMatrixCSC,
        op,
        v,
        i,
        j,
        part = 1
    )
    k = findindex(ext.cscmatrix, i, j)
    return if k > 0
        ext.cscmatrix.nzval[k] = op(ext.cscmatrix.nzval[k], v)
    else
        rawupdateindex!(ext.xmatrix, op, v, i, j)
    end
end

"""
    $(TYPEDSIGNATURES)
"""
function updateindex!(
        ext::GenericExtendableSparseMatrixCSC,
        op,
        v,
        i,
        j
    )
    k = findindex(ext.cscmatrix, i, j)
    return if k > 0
        ext.cscmatrix.nzval[k] = op(ext.cscmatrix.nzval[k], v)
    else
        updateindex!(ext.xmatrix, op, v, i, j)
    end
end
