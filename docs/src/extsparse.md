# Extendable matrices

## Abstract type
```@docs
ExtendableSparse.AbstractExtendableSparseMatrixCSC
```
## Implemented subtypes
```@docs
ExtendableSparse.GenericExtendableSparseMatrixCSC
ExtendableSparse.GenericMTExtendableSparseMatrixCSC
```

## Type aliases
```@docs
MTExtendableSparseMatrixCSC
STExtendableSparseMatrixCSC
ExtendableSparseMatrixCSC
ExtendableSparseMatrix
```


## Required methods
```@docs
SparseArrays.sparse
ExtendableSparse.rawupdateindex!
ExtendableSparse.flush!
ExtendableSparse.reset!
```

## AbstractSparseMatrixCSC interface
See [SparseArrways#395](https://github.com/JuliaSparse/SparseArrays.jl/pull/395)  for a discussion.


```@docs
SparseArrays.nnz
SparseArrays.nonzeros
SparseArrays.rowvals
SparseArrays.findnz
SparseArrays.dropzeros!
SparseArrays.getcolptr
SparseArrays.SparseMatrixCSC
Base.size
Base.eltype
Base.show
```

## Linear Algebra
```@docs
Base.:\
LinearAlgebra.ldiv!
LinearAlgebra.mul!
LinearAlgebra.norm
LinearAlgebra.opnorm
LinearAlgebra.cond
LinearAlgebra.issymmetric
```

## Algebraic operations
```@docs
Base.:+
Base.:-
Base.:*

```

## Handling of homogeneous Dirichlet BC
```@docs
mark_dirichlet
eliminate_dirichlet!
eliminate_dirichlet
```
