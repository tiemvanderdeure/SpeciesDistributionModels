module SpeciesDistributionModels

using MLJ 

import GLM, Tables, StatsBase, PrettyTables, Rasters, EvoTrees, DecisionTree

import CategoricalArrays.CategoricalArray

export SDMensemble, predict, sdm, select, machines, machine_keys

include("models.jl")
include("ensemble.jl")
include("predict.jl")

end


