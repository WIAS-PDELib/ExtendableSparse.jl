"""
    abstract type AbstractExtendableSparseMatrixCSC{Tv, Ti} <: AbstractSparseMatrixCSC{Tv, Ti} end

Abstract super type for extendable CSC matrices. It implements what is being discussed
as the "AbstractSparseMatrixCSC interfacee"

Subtypes must implement:
- SparseArrays.sparse  flush+return SparseMatrixCSC
- Constructor from SparseMatrixCSC
- rawupdateindex!
- reset!: empty all internals, just keep size 
- flush!: (re)build SparseMatrixCSC, incorporating new entries
"""
abstract type AbstractExtendableSparseMatrixCSC{Tv, Ti} <: AbstractSparseMatrixCSC{Tv, Ti} end

"""
    SparseArrays.sparse(A::AbstractExtendableSparseMatrixCSC)

Return `SparseMatrixCSC` which contains all matrix entries introduced so far.
"""
function SparseArrays.sparse(A::AbstractExtendableSparseMatrixCSC)
    throw(MethodError("Missing implementation of `sparse(::$(typeof(A)))`"))
end

"""
    rawupdateindex!(A::AbstractExtendableSparseMatrixCSC,op,v,i,j,part = 1)

Add or update entry of A: `A[i,j]=op(A[i,j],v)` without checking  if a zero
is inserted. The optional parameter part denotes the partition.
"""
function rawupdateindex!(A::AbstractExtendableSparseMatrixCSC, op, v, i, j, part = 1)
    throw(MethodError("Missing implementation of `rawupdateindex!(::$(typeof(A)),...)`"))
end

function flush!(A::AbstractExtendableSparseMatrixCSC)
    throw(MethodError("Missing implementation of `flush!(::$(typeof(A)))`"))
end

function reset!(A::AbstractExtendableSparseMatrixCSC)
    throw(MethodError("Missing implementation of `reset!(::$(typeof(A)))`"))
end


"""
    SparseArrays.nnz(ext::AbstractExtendableSparseMatrixCSC)

[`flush!`](@ref) and return number of nonzeros in ext.cscmatrix.
"""
SparseArrays.nnz(ext::AbstractExtendableSparseMatrixCSC) = nnz(sparse(ext))

"""
    SparseArrays.nonzeros(ext::AbstractExtendableSparseMatrixCSC) = nonzeros(sparse(ext))

[`flush!`](@ref) and return nonzeros in ext.cscmatrix.
"""
SparseArrays.nonzeros(ext::AbstractExtendableSparseMatrixCSC) = nonzeros(sparse(ext))

"""
$(TYPEDSIGNATURES)
"""
function SparseArrays.dropzeros!(ext::AbstractExtendableSparseMatrixCSC)
    return dropzeros!(sparse(ext))
end


"""
    Base.size(ext::AbstractExtendableSparseMatrixCSC)

Return size of matrix.
"""
Base.size(ext::AbstractExtendableSparseMatrixCSC) = size(ext.cscmatrix)


"""
    SparseArrays.rowvals(ext::AbstractExtendableSparseMatrixCSC)

[`flush!`](@ref) and return rowvals in ext.cscmatrix.
"""
SparseArrays.rowvals(ext::AbstractExtendableSparseMatrixCSC) = rowvals(sparse(ext))


"""
    SparseArrays.findnz(ext::AbstractExtendableSparseMatrixCSC)

[`flush!`](@ref) and return findnz(ext.cscmatrix).
"""
SparseArrays.findnz(ext::AbstractExtendableSparseMatrixCSC) = findnz(sparse(ext))


"""
    SparseArrays.getcolptr(ext::AbstractExtendableSparseMatrixCSC)

[`flush!`](@ref) and return colptr of  in ext.cscmatrix.
"""
SparseArrays.getcolptr(ext::AbstractExtendableSparseMatrixCSC) = getcolptr(sparse(ext))


"""
    Base.eltype(::AbstractExtendableSparseMatrixCSC{Tv, Ti})

Return element type.
"""
Base.eltype(::AbstractExtendableSparseMatrixCSC{Tv, Ti}) where {Tv, Ti} = Tv


"""
    SparseArrays.SparseMatrixCSC(A::AbstractExtendableSparseMatrixCSC)
Create SparseMatrixCSC from ExtendableSparseMatrix
"""
SparseArrays.SparseMatrixCSC(A::AbstractExtendableSparseMatrixCSC) = sparse(A)

"""
    Base.show(::IO, ::MIME"text/plain", ext::AbstractExtendableSparseMatrixCSC)

[`flush!`](@ref) and use the show method of SparseMatrixCSC to show the content.
"""
function Base.show(io::IO, ::MIME"text/plain", ext::AbstractExtendableSparseMatrixCSC)
    A = sparse(ext)
    xnnz = nnz(A)
    m, n = size(A)
    print(
        io,
        m,
        "×",
        n,
        " ",
        typeof(ext),
        " with ",
        xnnz,
        " stored ",
        xnnz == 1 ? "entry" : "entries"
    )

    if !haskey(io, :compact)
        io = IOContext(io, :compact => true)
    end

    return if !(m == 0 || n == 0 || xnnz == 0)
        print(io, ":\n")
        Base.print_array(IOContext(io), A)
    end
end


SparseArrays._checkbuffers(ext::AbstractExtendableSparseMatrixCSC) = SparseArrays._checkbuffers(sparse(ext))

"""
     Base.:\\(::AbstractExtendableSparseMatrixCSC, b)


[`\\`](@ref) for ExtendableSparse. It calls the LU factorization form Sparspak.jl, unless GPL components
are allowed  in the Julia sysimage and the floating point type of the matrix is Float64 or Complex64.
In that case, Julias standard `\` is called, which is realized via UMFPACK.
"""
function Base.:\(
        ext::AbstractExtendableSparseMatrixCSC{Tv, Ti},
        b::AbstractVector
    ) where {Tv, Ti}
    return sparspaklu(sparse(ext)) \ b
end


"""
     Base.:\\(Symmetric(::AbstractExtendableSparseMatrixCSC), b)

[`\\`](@ref) for Symmetric{ExtendableSparse}
"""
function Base.:\(
        symm_ext::Symmetric{Tm, T},
        b::AbstractVector
    ) where {Tm, Ti, T <: AbstractExtendableSparseMatrixCSC{Tm, Ti}}
    return Symmetric(sparse(symm_ext.data), Symbol(symm_ext.uplo)) \ b # no ldlt yet ...
end

"""
     Base.:\\(Hermitian(::AbstractExtendableSparseMatrixCSC), b)

[`\\`](@ref) for Hermitian{ExtendableSparse}
"""
function Base.:\(
        symm_ext::Hermitian{Tm, T},
        b::AbstractVector
    ) where {Tm, Ti, T <: AbstractExtendableSparseMatrixCSC{Tm, Ti}}
    return Hermitian(sparse(symm_ext.data), Symbol(symm_ext.uplo)) \ b # no ldlt yet ...
end

if USE_GPL_LIBS
    for (Tv) in (:Float64, :ComplexF64)
        @eval begin
            function Base.:\(
                    ext::AbstractExtendableSparseMatrixCSC{$Tv, Ti},
                    B::AbstractVector
                ) where {Ti}
                return sparse(ext) \ B
            end
        end

        @eval begin
            function Base.:\(
                    symm_ext::Symmetric{
                        $Tv,
                        AbstractExtendableSparseMatrixCSC{
                            $Tv,
                            Ti,
                        },
                    },
                    B::AbstractVector
                ) where {Ti}
                symm_csc = Symmetric(sparse(symm_ext.data), Symbol(symm_ext.uplo))
                return symm_csc \ B
            end
        end

        @eval begin
            function Base.:\(
                    symm_ext::Hermitian{
                        $Tv,
                        AbstractExtendableSparseMatrixCSC{
                            $Tv,
                            Ti,
                        },
                    },
                    B::AbstractVector
                ) where {Ti}
                symm_csc = Hermitian(sparse(symm_ext.data), Symbol(symm_ext.uplo))
                return symm_csc \ B
            end
        end
    end
end # USE_GPL_LIBS

"""
$(TYPEDSIGNATURES)

[`flush!`](@ref) and ldiv! with ext.cscmatrix
"""
function LinearAlgebra.ldiv!(r::AbstractArray, ext::AbstractExtendableSparseMatrixCSC, x::AbstractArray)
    return LinearAlgebra.ldiv!(r, sparse(ext), x)
end

"""
$(TYPEDSIGNATURES)

[`flush!`](@ref) and multiply with ext.cscmatrix
"""
function LinearAlgebra.mul!(r::AbstractVecOrMat, ext::AbstractExtendableSparseMatrixCSC, x::AbstractVecOrMat)
    return LinearAlgebra.mul!(r, sparse(ext), x)
end


# to resolve ambiguity
function LinearAlgebra.mul!(::SparseArrays.AbstractSparseMatrixCSC, ::ExtendableSparse.AbstractExtendableSparseMatrixCSC, ::LinearAlgebra.Diagonal)
    throw(MethodError("mul!(::AbstractSparseMatrixCSC, ::AbstractExtendableSparseMatrixCSC, ::Diagonal) is impossible"))
    return nothing
end

# to resolve ambiguity
function LinearAlgebra.mul!(::AbstractMatrix, ::ExtendableSparse.AbstractExtendableSparseMatrixCSC, ::LinearAlgebra.AbstractTriangular)
    throw(MethodError("mul!(::AbstractMatrix, ::AbstractExtendableSparseMatrixCSC, ::AbstractTriangular) is impossible"))
    return nothing
end


"""
$(TYPEDSIGNATURES)

[`flush!`](@ref) and calculate norm from cscmatrix
"""
function LinearAlgebra.norm(A::AbstractExtendableSparseMatrixCSC, p::Real = 2)
    return LinearAlgebra.norm(sparse(A), p)
end

"""
$(TYPEDSIGNATURES)

[`flush!`](@ref) and calculate opnorm from cscmatrix
"""
function LinearAlgebra.opnorm(A::AbstractExtendableSparseMatrixCSC, p::Real = 2)
    return LinearAlgebra.opnorm(sparse(A), p)
end

"""
$(TYPEDSIGNATURES)

[`flush!`](@ref) and calculate cond from cscmatrix
"""
function LinearAlgebra.cond(A::AbstractExtendableSparseMatrixCSC, p::Real = 2)
    return LinearAlgebra.cond(sparse(A), p)
end

"""
$(TYPEDSIGNATURES)

[`flush!`](@ref) and check for symmetry of cscmatrix
"""
function LinearAlgebra.issymmetric(A::AbstractExtendableSparseMatrixCSC)
    return LinearAlgebra.issymmetric(sparse(A))
end


"""
$(TYPEDSIGNATURES)
"""
function Base.:+(A::T, B::T) where {T <: AbstractExtendableSparseMatrixCSC}
    return T(sparse(A) + sparse(B))
end

"""
$(TYPEDSIGNATURES)
"""
function Base.:-(A::T, B::T) where {T <: AbstractExtendableSparseMatrixCSC}
    return T(sparse(A) - sparse(B))
end

"""
$(TYPEDSIGNATURES)
"""
function Base.:*(A::T, B::T) where {T <: AbstractExtendableSparseMatrixCSC}
    return T(sparse(A) * sparse(B))
end

"""
$(TYPEDSIGNATURES)
"""
function Base.:*(d::Diagonal, ext::T) where {T <: AbstractExtendableSparseMatrixCSC}
    return T(d * sparse(ext))
end

"""
$(TYPEDSIGNATURES)
"""
function Base.:*(ext::T, d::Diagonal) where {T <: AbstractExtendableSparseMatrixCSC}
    return T(sparse(ext) * d)
end


"""
$(TYPEDSIGNATURES)
"""
function Base.:+(ext::AbstractExtendableSparseMatrixCSC, csc::SparseMatrixCSC)
    return sparse(ext) + csc
end


"""
$(TYPEDSIGNATURES)
"""
function Base.:-(ext::AbstractExtendableSparseMatrixCSC, csc::SparseMatrixCSC)
    return sparse(ext) - csc
end

"""
$(TYPEDSIGNATURES)
"""
function Base.:-(csc::SparseMatrixCSC, ext::AbstractExtendableSparseMatrixCSC)
    return csc - sparse(ext)
end


"""
$(TYPEDSIGNATURES)
"""
function mark_dirichlet(A::AbstractExtendableSparseMatrixCSC; penalty = 1.0e20)
    return mark_dirichlet(sparse(A); penalty)
end

"""
$(TYPEDSIGNATURES)
"""
function eliminate_dirichlet(A::T, dirichlet) where {T <: AbstractExtendableSparseMatrixCSC}
    return T(eliminate_dirichlet(sparse(A), dirichlet))
end

"""
$(TYPEDSIGNATURES)
"""
function eliminate_dirichlet!(A::AbstractExtendableSparseMatrixCSC, dirichlet)
    eliminate_dirichlet!(sparse(A), dirichlet)
    return A
end
