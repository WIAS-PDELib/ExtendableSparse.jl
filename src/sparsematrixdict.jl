"""
    $(TYPEDEF)    

[`AbstractSparseMatrixExtension`](@ref)  extension where entries are organized as dictionary.
This is meant as an example implementation to show how a sparse matrix
extension could be implemented. As dictionary access tends to be slow, it
is not meant for general use.

An advantage of this format is the fact that it avoids to store a vector of the length of unknowns
indicating the first column indices, avoiding storage overhead during parallel assembly.

$(TYPEDFIELDS)
"""
mutable struct SparseMatrixDict{Tv, Ti} <: AbstractSparseMatrixExtension{Tv, Ti}
    """
    Number of rows
    """
    m::Ti

    """
    Number of columns
    """
    n::Ti

    """
    Dictionary with pairs of integers as keys containing values
    """
    values::Dict{Pair{Ti, Ti}, Tv}

    SparseMatrixDict{Tv, Ti}(m, n) where {Tv, Ti} = new(m, n, Dict{Pair{Ti, Ti}, Tv}())
end

"""
    $(TYPEDSIGNATURES)
"""
function SparseMatrixDict(Acsc::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    A = SparseMatrixDict{Tv, Ti}(size(Acsc)...)
    rows = rowvals(Acsc)
    vals = nonzeros(Acsc)
    m, n = size(Acsc)
    for j in 1:n
        for k in nzrange(Acsc, j)
            A[rows[k], j] = vals[k]
        end
    end
    return A
end

"""
    $(TYPEDSIGNATURES)
"""
function SparseMatrixDict(
        valuetype::Type{Tv}, indextype::Type{Ti}, m,
        n
    ) where {Tv, Ti <: Integer}
    return SparseMatrixDict{Tv, Ti}(m, n)
end


"""
    $(TYPEDSIGNATURES)
"""
SparseMatrixDict(valuetype::Type{Tv}, m, n) where {Tv} = SparseMatrixDict(Tv, Int, m, n)


"""
    $(TYPEDSIGNATURES)
"""
SparseMatrixDict(m, n) = SparseMatrixDict(Float64, m, n)

"""
    $(TYPEDSIGNATURES)
"""
function reset!(m::SparseMatrixDict{Tv, Ti}) where {Tv, Ti}
    return m.values = Dict{Pair{Ti, Ti}, Tv}()
end

"""
    $(TYPEDSIGNATURES)
"""
function Base.setindex!(m::SparseMatrixDict, v, i, j)
    return m.values[Pair(i, j)] = v
end

"""
    $(TYPEDSIGNATURES)
"""
function rawupdateindex!(m::SparseMatrixDict{Tv, Ti}, op, v, i, j) where {Tv, Ti}
    p = Pair(i, j)
    return m.values[p] = op(get(m.values, p, zero(Tv)), v)
end


"""
    $(TYPEDSIGNATURES)
"""
function updateindex!(m::SparseMatrixDict{Tv, Ti}, op, v, i, j) where {Tv, Ti}
    p = Pair(i, j)
    v1 = op(get(m.values, p, zero(Tv)), v)
    if !iszero(v1)
        m.values[p] = v1
    end
    return v1
end

"""
    $(TYPEDSIGNATURES)
"""
function Base.getindex(m::SparseMatrixDict{Tv}, i, j) where {Tv}
    return get(m.values, Pair(i, j), zero(Tv))
end

"""
    $(TYPEDSIGNATURES)
"""
Base.size(m::SparseMatrixDict) = (m.m, m.n)

"""
    $(TYPEDSIGNATURES)
"""
SparseArrays.nnz(m::SparseMatrixDict) = length(m.values)

"""
    $(TYPEDSIGNATURES)
"""
function SparseArrays.sparse(mat::SparseMatrixDict{Tv, Ti}) where {Tv, Ti}
    l = length(mat.values)
    I = Vector{Ti}(undef, l)
    J = Vector{Ti}(undef, l)
    V = Vector{Tv}(undef, l)
    i = 1
    for (p, v) in mat.values
        I[i] = first(p)
        J[i] = last(p)
        V[i] = v
        i = i + 1
    end
    @static if VERSION >= v"1.10"
        return SparseArrays.sparse!(I, J, V, size(mat)..., +)
    else
        return SparseArrays.sparse(I, J, V, size(mat)..., +)
    end
end

"""
    $(TYPEDSIGNATURES)
"""
function Base.:+(dictmatrix::SparseMatrixDict{Tv, Ti}, cscmatrix::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    lnew = length(dictmatrix.values)
    if lnew > 0
        (; colptr, nzval, rowval, m, n) = cscmatrix
        l = lnew + nnz(cscmatrix)

        I = Vector{Ti}(undef, l)
        J = Vector{Ti}(undef, l)
        V = Vector{Tv}(undef, l)
        i = 1
        for icsc in 1:(length(colptr) - 1)
            for j in colptr[icsc]:(colptr[icsc + 1] - 1)
                I[i] = rowval[j]
                J[i] = icsc
                V[i] = nzval[j]
                i = i + 1
            end
        end

        for (p, v) in dictmatrix.values
            I[i] = first(p)
            J[i] = last(p)
            V[i] = v
            i = i + 1
        end

        @assert l == i - 1
        @static if VERSION >= v"1.10"
            return SparseArrays.sparse!(I, J, V, m, n, +)
        else
            return SparseArrays.sparse(I, J, V, m, n, +)
        end
    end
    return cscmatrix
end

"""
    $(TYPEDSIGNATURES)
"""
function Base.sum(dictmatrices::Vector{SparseMatrixDict{Tv, Ti}}, cscmatrix::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    lnew = sum(m -> length(m.values), dictmatrices)
    if lnew > 0
        (; colptr, nzval, rowval, m, n) = cscmatrix
        l = lnew + nnz(cscmatrix)
        I = Vector{Ti}(undef, l)
        J = Vector{Ti}(undef, l)
        V = Vector{Tv}(undef, l)
        i = 1

        for icsc in 1:(length(colptr) - 1)
            for j in colptr[icsc]:(colptr[icsc + 1] - 1)
                I[i] = icsc
                J[i] = rowval[j]
                V[i] = nzval[j]
                i = i + 1
            end
        end

        ip = 1
        for m in dictmatrices
            for (p, v) in m.values
                I[i] = first(p)
                J[i] = last(p)
                V[i] = v
                i = i + 1
            end
            ip = ip + 1
        end
        @static if VERSION >= v"1.10"
            return SparseArrays.sparse!(I, J, V, m, n, +)
        else
            return SparseArrays.sparse(I, J, V, m, n, +)
        end
    end
    return cscmatrix
end
