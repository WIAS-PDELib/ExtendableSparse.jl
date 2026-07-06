module ExtendableSparseLinearSolveExt
using LinearSolve: LinearSolve, LinearProblem, solve!, init
import ExtendableSparse: LinearSolvePreconBuilder
import LinearAlgebra
using SparseArrays: AbstractSparseMatrixCSC


struct LinearSolvePrecon{T}
    cache::T
end

function LinearSolvePrecon(A, method::LinearSolve.AbstractFactorization)
    pr = LinearProblem(A, zeros(eltype(A), size(A, 1)))
    return LinearSolvePrecon(init(pr, method))
end

function LinearAlgebra.ldiv!(u, P::LinearSolvePrecon, b)
    P.cache.b = b
    sol = solve!(P.cache)
    return copyto!(u, sol.u)
end

(b::LinearSolvePreconBuilder)(A::AbstractSparseMatrixCSC, p) = (LinearSolvePrecon(A, b.method), LinearAlgebra.I)

end
