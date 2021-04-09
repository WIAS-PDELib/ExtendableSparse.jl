"""
  Abstract type for an extendabe preconditoioner working together
  with ExtandableSparseMatrix. 
  Still beta, so not in the published documentation.
  
  Any such preconditioner should have the following fields
````
  extmatrix
  phash
````
and  methods
````
  update!(precon)
````   
  The idea is that, depending if the matrix pattern has changed, 
  different steps are needed to update the preconditioner.

  Moreover, they have the ExtendableSparseMatrix as a field, ensuring 
  consistency after construction.
"""
abstract type AbstractExtendablePreconditioner{Tv,Ti} end

need_symbolic_update(precon::AbstractExtendablePreconditioner)=precon.extmatrix.phash!=precon.phash

include("jacobi.jl")
include("ilu0.jl")
include("parallel_jacobi.jl")
include("default_lu.jl")

