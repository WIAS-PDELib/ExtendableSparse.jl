# Changelog

## [2.0.0] - Planned

### Breaking
- remove solver + precon API which is not based on precs or directly overloading `\`.
  Fully rely on LinearSolve (besides `\`)
- Move AMGBuilder, ILUZeroBuilder to the correspondig packages (depending on the PRs)
- remove "old" SparseMatrixLNK (need to benchmark before)

## [1.6.0] - WIP
- Support precs API of LinearSolve.jl

## [1.5.3] - 2024-10-07
- Moved repo to WIAS-PDELib organization

## [1.5.2] - 2024-10-07

- Bump version of AMGCLWrap

## [1.5.1] - 2024-07-22

- Update changelog
- Remove OhMyThreads dependency

## [1.5.0] - 2024-07-17

- First multithreaded version
## [1.4.1] - 2024-06-23

- Experimenta multihreading

## [1.4.0] - 2024-01-25

- Add dirichlet node elimination

## [1.3.1] - 2024-01-17
* AMGCLWrap extension
* Deprecate AMGPreconditioner
* introduce RS, SA amg preconditioner from AlgebraicMultigrid

## [1.2.1] - 2023-12-16

- Extensions (#27)

Introducd 1.9 extensions

## [1.1.0] - 2023-05-03

- Block preconditioning

* Generic block preconditioner
* Removed element types from factorization wrapper

## [1.0.1] - 2023-02-21

- Implement AbstractSparseMatrixCSC interface.
- Remove extension of LinearSolve methodsa

