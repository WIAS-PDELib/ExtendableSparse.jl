"""
        LinearSolvePreconBuilder(; method=UMFPACKFactorization())

Return callable object constructing a formal left preconditioner from a sparse LU factorization using LinearSolve
as the `precs` parameter for a  [`BlockPreconBuilder`](@ref)  or  iterative methods wrapped by LinearSolve.jl.
"""
Base.@kwdef struct LinearSolvePreconBuilder
    method = UMFPACKFactorization()
end
(::LinearSolvePreconBuilder)(A, p) = error("import LinearSolve in order to use LinearSolvePreconBuilder")


"""
    JacobiPreconBuilder()

Return callable object constructing a left Jacobi preconditioner
to be passed as the `precs` parameter to iterative methods wrapped by LinearSolve.jl.
"""
struct JacobiPreconBuilder end
(::JacobiPreconBuilder)(A::AbstractSparseMatrixCSC, p) = (JacobiPreconditioner(SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A), nonzeros(A))), LinearAlgebra.I)


"""
    ILUZeroPreconBuilder(;blocksize=1)

Return callable object constructing a left zero fill-in ILU preconditioner 
using [ILUZero.jl](https://github.com/mcovalt/ILUZero.jl)
"""
Base.@kwdef mutable struct ILUZeroPreconBuilder
    blocksize::Int = 1
end

struct ILUBlockPrecon{N, NN, Tv, Ti}
    ilu0::ILUZero.ILU0Precon{SMatrix{N, N, Tv, NN}, Ti, SVector{N, Tv}}
end

function LinearAlgebra.ldiv!(
        Y::Vector{Tv},
        A::ILUBlockPrecon{N, NN, Tv, Ti},
        B::Vector{Tv}
    ) where {N, NN, Tv, Ti}
    BY = reinterpret(SVector{N, Tv}, Y)
    BB = reinterpret(SVector{N, Tv}, B)
    ldiv!(BY, A.ilu0, BB)
    return Y
end

"""
    pointblock(matrix,blocksize)

Create a pointblock matrix.
"""
function pointblock(A0::ExtendableSparseMatrixCSC{Tv, Ti}, blocksize) where {Tv, Ti}
    A = SparseMatrixCSC(A0)
    colptr = A.colptr
    rowval = A.rowval
    nzval = A.nzval
    n = A.n
    block = zeros(Tv, blocksize, blocksize)
    nblock = n ÷ blocksize
    b = SMatrix{blocksize, blocksize}(block)
    Tb = typeof(b)
    Ab = ExtendableSparseMatrixCSC{Tb, Ti}(nblock, nblock)


    for i in 1:n
        for k in colptr[i]:(colptr[i + 1] - 1)
            j = rowval[k]
            iblock = (i - 1) ÷ blocksize + 1
            jblock = (j - 1) ÷ blocksize + 1
            ii = (i - 1) % blocksize + 1
            jj = (j - 1) % blocksize + 1
            block[ii, jj] = nzval[k]
            rawupdateindex!(Ab, +, SMatrix{blocksize, blocksize}(block), iblock, jblock)
            block[ii, jj] = zero(Tv)
        end
    end
    return flush!(Ab)
end

function (b::ILUZeroPreconBuilder)(A0, p)
    A = SparseMatrixCSC(size(A0)..., getcolptr(A0), rowvals(A0), nonzeros(A0))
    return if b.blocksize == 1
        (ILUZero.ilu0(A), LinearAlgebra.I)
    else
        (ILUBlockPrecon(ILUZero.ilu0(pointblock(A, b.blocksize), SVector{b.blocksize, eltype(A)})), LinearAlgebra.I)
    end
end


"""
    ILUTPreconBuilder(; droptol=0.1)

Return callable object constructing a left ILUT preconditioner 
using [IncompleteLU.jl](https://github.com/haampie/IncompleteLU.jl)
"""
Base.@kwdef struct ILUTPreconBuilder
    droptol::Float64 = 0.1
end
(::ILUTPreconBuilder)(A, p) = error("import IncompleteLU.jl in order to use ILUTBuilder")


mutable struct BlockPreconditioner
    A::AbstractMatrix
    factorization
    partitioning::Union{Nothing, Vector{AbstractVector}}
    facts::Vector
    function BlockPreconditioner(A; partitioning = nothing, factorization = nothing)
        p = new()
        p.A = A
        p.partitioning = partitioning
        p.factorization = factorization
        update!(p)
        return p
    end
end


"""
     BlockPreconditioner(;partitioning, factorization)
    
Create a block preconditioner from partition of unknowns given by `partitioning`, a vector of AbstractVectors describing the
indices of the partitions of the matrix. For a matrix of size `n x n`, e.g. partitioning could be `[ 1:n÷2, (n÷2+1):n]`
or [ 1:2:n, 2:2:n].
Factorization is a callable (Function or struct) which allows to create a factorization (with `ldiv!` methods) from a submatrix of A.
"""
function BlockPreconditioner end

"""
    allow_views(::preconditioner_type)

Factorizations on matrix partitions within a block preconditioner may or may not work with array views.
E.g. the umfpack factorization cannot work with views, while ILUZeroPreconditioner can.
 Implementing a method for `allow_views` returning `false` resp. `true` allows to dispatch to the proper case.
"""
allow_views(::Any) = false


function update!(precon::BlockPreconditioner)
    flush!(precon.A)
    nall = sum(length, precon.partitioning)
    n = size(precon.A, 1)
    if nall != n
        @warn "sum(length,partitioning)=$(nall) but n=$(n)"
    end

    if isnothing(precon.partitioning)
        partitioning = [1:n]
    end

    np = length(precon.partitioning)
    precon.facts = Vector{Any}(undef, np)
    return Threads.@threads for ipart in 1:np
        factorization = deepcopy(precon.factorization)
        AP = precon.A[precon.partitioning[ipart], precon.partitioning[ipart]]
        FP = factorization(AP)
        precon.facts[ipart] = FP
    end
end


function LinearAlgebra.ldiv!(p::BlockPreconditioner, v)
    partitioning = p.partitioning
    facts = p.facts
    np = length(partitioning)

    if allow_views(p.factorization)
        Threads.@threads for ipart in 1:np
            ldiv!(facts[ipart], view(v, partitioning[ipart]))
        end
    else
        Threads.@threads for ipart in 1:np
            vv = v[partitioning[ipart]]
            ldiv!(facts[ipart], vv)
            view(v, partitioning[ipart]) .= vv
        end
    end
    return v
end

function LinearAlgebra.ldiv!(u, p::BlockPreconditioner, v)
    partitioning = p.partitioning
    facts = p.facts
    np = length(partitioning)
    if allow_views(p.factorization)
        Threads.@threads for ipart in 1:np
            ldiv!(view(u, partitioning[ipart]), facts[ipart], view(v, partitioning[ipart]))
        end
    else
        Threads.@threads for ipart in 1:np
            uu = u[partitioning[ipart]]
            ldiv!(uu, facts[ipart], v[partitioning[ipart]])
            view(u, partitioning[ipart]) .= uu
        end
    end
    return u
end

Base.eltype(p::BlockPreconditioner) = eltype(p.facts[1])


"""
     BlockPreconBuilder(;precs=UMFPACKPreconBuilder(),  
                         partitioning = A -> [1:size(A,1)]

Return callable object constructing a left block Jacobi preconditioner 
from partition of unknowns.

- `partitioning(A)`  shall return a vector of AbstractVectors describing the indices of the partitions of the matrix. 
  For a matrix of size `n x n`, e.g. partitioning could be `[ 1:n÷2, (n÷2+1):n]` or [ 1:2:n, 2:2:n].

- `precs(A,p)` shall return a left precondioner for a matrix block.
"""
Base.@kwdef mutable struct BlockPreconBuilder
    precs = UMFPACKPreconBuilder()
    partitioning = A -> [1:size(A, 1)]
end

function (blockprecs::BlockPreconBuilder)(A, p)
    (; precs, partitioning) = blockprecs
    factorization = A -> precs(A, p)[1]
    bp = BlockPreconditioner(A; partitioning = partitioning(A), factorization)
    return (bp, LinearAlgebra.I)
end

"""
    Allow array for precs => different precoms
"""


mutable struct _JacobiPreconditioner{Tv}
    invdiag::Vector{Tv}
end

function jacobi(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    invdiag = Array{Tv, 1}(undef, A.n)
    n = A.n
    @inbounds for i in 1:n
        invdiag[i] = one(Tv) / A[i, i]
    end
    return _JacobiPreconditioner(invdiag)
end

function jacobi!(p::_JacobiPreconditioner{Tv}, A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    n = A.n
    @inbounds for i in 1:n
        p.invdiag[i] = one(Tv) / A[i, i]
    end
    return p
end

mutable struct JacobiPreconditioner
    A::AbstractMatrix
    factorization::Union{_JacobiPreconditioner, Nothing}
    function JacobiPreconditioner(A)
        p = new()
        p.A = A
        p.factorization = nothing
        update!(p)
        return p
    end
end

function LinearAlgebra.ldiv!(u, p::_JacobiPreconditioner, v)
    n = length(p.invdiag)
    for i in 1:n
        @inbounds u[i] = p.invdiag[i] * v[i]
    end
    return u
end

LinearAlgebra.ldiv!(p::_JacobiPreconditioner, v) = ldiv!(v, p, v)


"""
```
JacobiPreconditioner()
JacobiPreconditioner(matrix)
```

Jacobi preconditioner.
"""
function JacobiPreconditioner end

function update!(p::JacobiPreconditioner)
    flush!(p.A)
    Tv = eltype(p.A)
    p.factorization = jacobi(SparseMatrixCSC(p.A))
    return p
end

LinearAlgebra.ldiv!(u, fact::JacobiPreconditioner, v) = ldiv!(u, fact.factorization, v)
LinearAlgebra.ldiv!(fact::JacobiPreconditioner, v) = ldiv!(fact.factorization, v)

allow_views(::JacobiPreconditioner) = true
allow_views(::Type{JacobiPreconditioner}) = true


"""
    SchurComplementPreconditioner{AT, ST}

Preconditioner for saddle-point problems of the form:
```
[ A  B ]
[ Bᵀ 0 ]
```

# Fields
- `partitions`: Vector of two ranges defining matrix partitioning
- `A_fac::AT`: Factorization of the A-block (top-left)
- `S_fac::ST`: Factorization of the Schur complement (approximation of -BᵀA⁻¹B)
"""
struct SchurComplementPreconditioner{AT, ST}
    partitions
    A_fac::AT
    S_fac::ST
end


"""
    SchurComplementPreconBuilder(dofs_first_block, A_factorization, S_factorization = lu)

Factory function creating preconditioners for saddle-point problems.

# Arguments
- `dofs_first_block`: Number of degrees of freedom in the first block (size of A matrix)
- `A_factorization`: Factorization method for the A-block (e.g., `ilu0, Diagonal, lu`)
- `S_factorization`: Factorization method for the Schur complement (defaults to `lu`)

Returns a function `prec(M, p)` that creates a `SchurComplementPreconditioner`.
"""
function SchurComplementPreconBuilder(dofs_first_block, A_factorization, S_factorization = lu; verbosity = 0)

    # this is the resulting preconditioner
    function prec(M)

        # We have
        # M = [ A  B ] → n1 dofs
        #     [ Bᵀ 0 ] → n2 dofs

        n1 = dofs_first_block
        n2 = size(M, 1) - n1

        # I see no other way than creating this expensive copy
        A = M[1:n1, 1:n1]
        B = M[1:n1, (n1 + 1):end]

        # first factorization
        A_fac = A_factorization(A)

        verbosity > 0 && @info "SchurComplementPreconBuilder: A ($n1×$n1) is factorized"

        # compute the Schur Matrix
        # S ≈ -BᵀA⁻¹B
        # we use the diagonal of A: this is _very_ performant and creates a sparse result
        S = - B' * (Diagonal(A) \ B)

        verbosity > 0 && @info "SchurComplementPreconBuilder: S ($n2×$n2) is computed"

        # factorize S
        S_fac = S_factorization(S)

        verbosity > 0 && @info "SchurComplementPreconBuilder: S ($n2×$n2) is factorized"

        return SchurComplementPreconditioner([1:n1, (n1 + 1):(n1 + n2)], A_fac, S_fac)
    end

    return prec
end


function LinearAlgebra.ldiv!(u, p::SchurComplementPreconditioner, v)

    (part1, part2) = p.partitions
    @views ldiv!(u[part1], p.A_fac, v[part1])
    @views ldiv!(u[part2], p.S_fac, v[part2])

    return u
end

function LinearAlgebra.ldiv!(p::SchurComplementPreconditioner, v)

    (part1, part2) = p.partitions
    @views ldiv!(p.A_fac, v[part1])
    @views ldiv!(p.S_fac, v[part2])

    return v
end


struct FullSchurComplementPreconditioner{AT, ST, BT}
    partitions
    A_fac::AT
    S_fac::ST
    B::BT
    Av1_cache::Vector{Float64}
    BSv2_cache::Vector{Float64}
    BSBAv1_cache::Vector{Float64}
    ABSBAv1_cache::Vector{Float64}
    ABSv2_cache::Vector{Float64}
    Sv2_cache::Vector{Float64}
    BAv1_cache::Vector{Float64}
    SBAv1_cache::Vector{Float64}
end


function FullSchurComplementPreconBuilder(dofs_first_block, A_factorization, S_factorization = lu; verbosity = 0)

    # this is the resulting preconditioner
    function prec(M)

        # We have
        # M = [ A  B ] → n1 dofs
        #     [ Bᵀ 0 ] → n2 dofs

        n1 = dofs_first_block
        n2 = size(M, 1) - n1

        # I see no other way than creating this expensive copy
        A = M[1:n1, 1:n1]
        B = M[1:n1, (n1 + 1):end]

        # first factorization
        A_fac = A_factorization(A)

        verbosity > 0 && @info "SchurComplementPreconBuilder: A ($n1×$n1) is factorized"

        # compute dense (!) the Schur Matrix
        # S ≈ -BᵀA⁻¹B
        # S = zeros(n2, n2)
        # for col in 1:n2
        #     @show col
        #     @views S[:, col] = B' * (A_fac \ Vector(B[:, col]))
        # end
        # S .*= -1.0

        S = - B' * (Diagonal(A) \ B)

        # S = zeros(n2, n2)

        # function col_loop(col_chunk)
        #     # sigh, some factorizations (like ilu0) store internal buffers. Hence, every thread needs a copy.
        #     local A_fac_copy = deepcopy(A_fac)

        #     # local buffers and result
        #     local ldiv_buffer = zeros(n1)
        #     local Bcol_buffer = zeros(n1)

        #     for (icol, col) in enumerate(col_chunk)
        #         @info "col $icol of $(length(col_chunk))"
        #         Bcol = B[:, col] # B is very sparse: this copy is fine?
        #         Bcol_buffer .= 0.0
        #         Bcol_buffer[Bcol.nzind] = Bcol.nzval
        #         @views ldiv!(ldiv_buffer, A_fac_copy, Bcol_buffer)
        #         @views mul!(S[:, col], B', ldiv_buffer, -1.0, 0.0)
        #     end
        #     return nothing
        # end

        # # split columns into chunks and assemble S
        # col_chunks = chunks(1:n2, n = Threads.nthreads())
        # tasks = map(col_chunks) do col_chunk
        #     @info "start thread"
        #     Threads.@spawn col_loop(col_chunk)
        # end

        # # then S is ready
        # fetch.(tasks)

        verbosity > 0 && @info "SchurComplementPreconBuilder: S ($n2×$n2) is computed"

        # factorize S
        S_fac = S_factorization(S)

        verbosity > 0 && @info "SchurComplementPreconBuilder: S ($n2×$n2) is factorized"

        return FullSchurComplementPreconditioner(
            [1:n1, (n1 + 1):(n1 + n2)],
            A_fac,
            S_fac,
            B,
            zeros(n1),
            zeros(n1),
            zeros(n1),
            zeros(n1),
            zeros(n1),
            zeros(n2),
            zeros(n2),
            zeros(n2),
        )
    end

    return prec
end


function LinearAlgebra.ldiv!(u, p::FullSchurComplementPreconditioner, v)

    # M⁻¹ =[ A⁻¹ + A⁻¹BS⁻¹BᵀA⁻¹   -A⁻¹BS⁻¹ ]
    #      [ -S⁻¹BᵀA⁻¹            S⁻¹      ]

    (part1, part2) = p.partitions
    @views v1 = v[part1]
    @views v2 = v[part2]

    (; A_fac, S_fac, B, Av1_cache, Sv2_cache, BAv1_cache, BSv2_cache, BSBAv1_cache, ABSBAv1_cache, ABSBAv1_cache, ABSv2_cache, SBAv1_cache) = p

    # A_fac \ v1
    @showtime ldiv!(Av1_cache, A_fac, v1)

    # S_fac \ v2
    @showtime ldiv!(Sv2_cache, S_fac, v2)

    # B' * (Av1_cache)
    @showtime mul!(BAv1_cache, B', Av1_cache)

    # S_fac \ BAv1_cache
    @showtime ldiv!(SBAv1_cache, S_fac, BAv1_cache)

    # B * Sv2_cache)
    @showtime mul!(BSv2_cache, B, Sv2_cache)

    # B * SBAv1_cache
    @showtime mul!(BSBAv1_cache, B, SBAv1_cache)

    # A_fac \ BSBAv1_cache
    @showtime ldiv!(ABSBAv1_cache, A_fac, BSBAv1_cache)

    # A_fac \ BSv2_cache
    @showtime ldiv!(ABSv2_cache, A_fac, BSv2_cache)

    @showtime @views u[part1] .= Av1_cache .+ ABSBAv1_cache .- ABSv2_cache
    @showtime @views u[part2] .= Sv2_cache .- SBAv1_cache
    return u
end

# function LinearAlgebra.ldiv!(p::FullSchurComplementPreconditioner, v)

#     (part1, part2) = p.partitions
#     @views ldiv!(p.A_fac, v[part1])
#     @views ldiv!(p.S_fac, v[part2])

#     return v
# end
