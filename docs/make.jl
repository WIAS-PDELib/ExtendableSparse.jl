using Documenter, ExtendableSparse, AlgebraicMultigrid, IncompleteLU, Sparspak, LinearAlgebra, SparseArrays, Base, InteractiveUtils

function mkdocs()
    return makedocs(;
        sitename = "ExtendableSparse.jl",
        modules = [ExtendableSparse],
        doctest = false,
        warnonly = true,
        clean = false,
        authors = "J. Fuhrmann",
        repo = "https://github.com/WIAS-PDELib/ExtendableSparse.jl",
        pages = [
            "Home" => "home.md",
            "example.md",
            "extsparse.md",
            "extensions.md",
            "linearsolve.md",
            "misc.md",
            "index.md",
        ]
    )
end

mkdocs()

deploydocs(; repo = "github.com/WIAS-PDELib/ExtendableSparse.jl.git")
