mutable struct LUFactorization{Tv,Ti} <: AbstractLUFactorization{Tv,Ti}
    A::Union{Nothing,ExtendableSparseMatrix{Tv,Ti}}
    fact::Union{Nothing,SuiteSparse.UMFPACK.UmfpackLU{Tv,Ti}}
    phash::UInt64
end


"""
```
LUFactorization()
LUFactorization(matrix)
```
        
Default Julia LU Factorization based on umfpack.
"""
LUFactorization()=LUFactorization{Float64,Int64}(nothing,nothing,0)


function update!(lufact::LUFactorization)
    flush!(lufact.A)
    if lufact.A.phash!=lufact.phash
        lufact.fact=lu(lufact.A.cscmatrix)
        lufact.phash=lufact.A.phash
    else
        lufact.fact=lu!(lufact.fact,lufact.A.cscmatrix)
    end
    lufact
end

