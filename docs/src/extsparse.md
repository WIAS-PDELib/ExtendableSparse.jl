# Extendable matrices
The type hierarchy of extendable matrices in this package is as follows:

[`ExtendableSparse.AbstractExtendableSparseMatrixCSC`](@ref) `<: SparseArrays.AbstractSparseMatrixCSC <: SparseArrays.AbstractSparseMatrix <: AbstractMatrix`

The package defines two [subtypes](#Subtypes-of-AbstractExtendableSparseMatrixCSC) of [`ExtendableSparse.AbstractExtendableSparseMatrixCSC`](@ref) which are parametrized by types of [extension matrices](/extensions/#Matrix-extensions):
- [`ExtendableSparse.GenericExtendableSparseMatrixCSC`](@ref) for single threaded assembly
- [`ExtendableSparse.GenericMTExtendableSparseMatrixCSC`](@ref) for multithreaded assembly

User facing defaults are defined by [type aliases](#Type-aliases):
- `const MTExtendableSparseMatrixCSC = GenericMTExtendableSparseMatrixCSC{SparseMatrixDILNKC}`
- `const STExtendableSparseMatrixCSC = GenericExtendableSparseMatrixCSC{SparseMatrixLNK}`
- `const ExtendableSparseMatrixCSC = STExtendableSparseMatrixCSC`
- `const ExtendableSparseMatrix = ExtendableSparseMatrixCSC`

## Abstract type
```@docs
ExtendableSparse.AbstractExtendableSparseMatrixCSC
```
## Subtypes of AbstractExtendableSparseMatrixCSC
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
