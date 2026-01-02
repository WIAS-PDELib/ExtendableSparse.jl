# Linear System solution

## The `\` operator
The packages overloads `\` for the ExtendableSparseMatrix type.
The following example uses [`fdrand`](@ref) to create a test matrix and solve
the corresponding linear system of equations.

```@example
using ExtendableSparse: ExtendableSparseMatrix, fdrand
A = fdrand(10, 10, 10; matrixtype = ExtendableSparseMatrix)
x = ones(1000)
b = A * x
y = A \ b
sum(y)
```

This works as well for number types besides `Float64` and related, in this case,
by default a LU factorization based on Sparspak is used.

```@example
using ExtendableSparse: ExtendableSparseMatrix, fdrand
using MultiFloats
A = fdrand(Float64x2, 10, 10, 10; matrixtype = ExtendableSparseMatrix)
x = ones(Float64x2,1000)
b = A * x
y = A \ b
sum(y)
```

## Solving with LinearSolve.jl

Starting with version 0.9.6, ExtendableSparse is compatible
with [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl).
Since version 0.9.7, this is facilitated via the
AbstractSparseMatrixCSC interface. 


The same problem can be solved via `LinearSolve.jl`:

```@example
using ExtendableSparse: ExtendableSparseMatrix, fdrand
using LinearSolve
A = fdrand(10, 10, 10; matrixtype = ExtendableSparseMatrix)
x = ones(1000)
b = A * x
y = solve(LinearProblem(A, b)).u
sum(y)
```

```@example
using ExtendableSparse: ExtendableSparseMatrix, fdrand
using LinearSolve
using MultiFloats
A = fdrand(Float64x2, 10, 10, 10; matrixtype = ExtendableSparseMatrix)
x = ones(Float64x2,1000)
b = A * x
y = solve(LinearProblem(A, b), SparspakFactorization()).u
sum(y)
```

## Preconditioned Krylov solvers  with LinearSolve.jl

Since version 1.6, preconditioner constructors can be passed to iterative solvers via the [`precs`
keyword argument](https://docs.sciml.ai/LinearSolve/stable/basics/Preconditioners/#prec)
to the iterative solver specification.

```@example
using ExtendableSparse: ExtendableSparseMatrix, fdrand
using LinearSolve 
using ExtendableSparse: ILUZeroPreconBuilder
A = fdrand(10, 10, 10; matrixtype = ExtendableSparseMatrix)
x = ones(1000)
b = A * x
y = LinearSolve.solve(LinearProblem(A, b), 
                      KrylovJL_CG(; precs=ILUZeroPreconBuilder())).u
sum(y)
```

## Available preconditioners
ExtendableSparse provides constructors for preconditioners which can be used as `precs`.
These generally return a tuple `(Pl,I)` of a left preconditioner and a trivial right preconditioner.

ExtendableSparse has a number of package extensions which construct preconditioners
from some other packages. In the future, these packages may provide this functionality on their own.

```@docs
ExtendableSparse.ILUZeroPreconBuilder
ExtendableSparse.ILUTPreconBuilder
```

In addition, ExtendableSparse implements some preconditioners:

```@docs
ExtendableSparse.JacobiPreconBuilder
```

LU factorizations of  matrices from previous iteration steps may be good
preconditioners for Krylov solvers called during a nonlinear solve via
Newton's method. For this purpose, ExtendableSparse provides a preconditioner constructor
which wraps sparse LU factorizations  supported by LinearSolve.jl
```@docs
ExtendableSparse.LinearSolvePreconBuilder
```


Block preconditioner constructors are provided as well
```@docs
ExtendableSparse.BlockPreconBuilder
```


The example beloww shows how to create a  block Jacobi preconditioner where the blocks are defined by even and odd
degrees of freedom, and the diagonal blocks are solved using UMFPACK.
```@example
using LinearSolve 
using ExtendableSparse: LinearSolvePreconBuilder, BlockPreconBuilder, ExtendableSparseMatrix, fdrand
A = fdrand(10, 10, 10; matrixtype = ExtendableSparseMatrix)
x = ones(1000)
b = A * x
partitioning=A->[1:2:size(A,1), 2:2:size(A,1)]
umfpackprecs=LinearSolvePreconBuilder(UMFPACKFactorization())
blockprecs=BlockPreconBuilder(;precs=umfpackprecs, partitioning)
y = LinearSolve.solve(LinearProblem(A, b), KrylovJL_CG(; precs=blockprecs)).u
sum(y)
```
`umpfackpreconbuilder` e.g. could be replaced by `SmoothedAggregationPreconBuilder()`. Moreover, this approach
works for any `AbstractSparseMatrixCSC`.

