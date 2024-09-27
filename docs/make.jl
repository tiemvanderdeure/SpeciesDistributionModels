using Documenter, DocumenterVitepress
using SpeciesDistributionModels

makedocs(;
    modules=[SpeciesDistributionModels],
    authors= "Tiem van der Deure <tvd@sund.ku.dk>, Rafael Schouten <rafaelschouten@gmail.com>",
    sitename="SpeciesDistributionModels.jl",
    format=DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/tiemvanderdeure/SpeciesDistributionModels.jl",
        devurl = "dev",
        devbranch = "master",
    ),
    warnonly = true,
    pages = [
        "Home" => "index.md",
        "Getting started" => "example.md",
        "API" => "api.md"
        ]
)

deploydocs(; 
    repo="github.com/tiemvanderdeure/SpeciesDistributionModels.jl",
    target = "build", # this is where Vitepress stores its output
    branch = "gh-pages",
    devbranch = "master",
    push_preview = true
)