using LinearAlgebra
using SparseArrays
using ExtendableSparse
using Printf
using BenchmarkTools

using MultiFloats
using ForwardDiff
using ExplicitImports

@testset "ExplicitImports" begin
    @test ExplicitImports.check_no_implicit_imports(ExtendableSparse) === nothing
    @test ExplicitImports.check_no_stale_explicit_imports(ExtendableSparse) === nothing
end

@testset "Parallel" begin
    include("test_parallel.jl")

    for Tm in [STExtendableSparseMatrixCSC, MTExtendableSparseMatrixCSC, ExtendableSparseMatrix]
        for N in [10000, 20000]
            test_parallel.test_correctness_build_seq(N, Tm, dim = 2)
        end
    end

    for Tm in [MTExtendableSparseMatrixCSC]
        for N in [10000, 20000]
            test_parallel.test_correctness_update(N, Tm, dim = 2)
            test_parallel.test_correctness_build(N, Tm, dim = 2)
            test_parallel.test_correctness_mul(N, Tm, dim = 2)
        end
    end
end

@testset "Constructors" begin
    include("test_constructors.jl")
end

@testset "Copy-Methods" begin
    include("test_copymethods.jl")
end

@testset "Updates" begin
    include("test_updates.jl")
end

@testset "Assembly" begin
    include("test_assembly.jl")
end

@testset "Construction timings" begin
    include("test_timings.jl")
end

@testset "Operations" begin
    include("test_operations.jl")
end

@testset "fdrand" begin
    include("test_fdrand.jl")
end

@testset "Backslash" begin
    include("test_backslash.jl")
end

@testset "Dirichlet" begin
    include("test_dirichlet.jl")
end

@testset "LinearSolve" begin
    include("test_linearsolve.jl")
end

@testset "Symmetric" begin
    include("test_symmetric.jl")
end

@testset "block" begin
    include("test_block.jl")
end
