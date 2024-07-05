using SpeciesDistributionModels, MLJBase, MLJModels, Tables
import SpeciesDistributionModels as SDM
using StableRNGs, Distributions, Test
using Makie

rng = StableRNG(0)
#using Random; rng = Random.GLOBAL_RNG
# some mock data
n = 100
backgrounddata = (a = rand(rng, n), b = rand(rng, n), c = rand(rng, n))
presencedata = (a = rand(rng, n), b = rand(rng, n).^2, c = sqrt.(rand(rng, n)))

@testset "SpeciesDistributionModels.jl" begin
    ## data
    data = sdmdata(presencedata, backgrounddata; resampler = CV(nfolds = 5, shuffle = true))
    # alternative sdm method
    x = map(presencedata, backgrounddata) do p, b
        [p; b]
    end
    y = [trues(Tables.rowcount(presencedata)); falses(Tables.rowcount(backgrounddata))]
    data2 = sdmdata(x, y)

    ## ensemble
    models = (
        rf = SDM.random_forest(; rng),
        rf2 = OneHotEncoder() |> SDM.random_forest(; max_depth = 3, rng),
        lm = SDM.linear_model(),
        brt = SDM.boosted_regression_tree(; rng)
    )

    ensemble = sdm(data, models;
        threaded = false
    )

    evaluation = SDM.evaluate(ensemble; validation = (presencedata, backgrounddata))
    evaluation2 = SDM.evaluate(ensemble)
    @test evaluation isa SDM.SDMensembleEvaluation
    @test evaluation[1] isa SDM.SDMgroupEvaluation
    @test evaluation[1][1] isa SDM.SDMmachineEvaluation
    @test SDM.measures(evaluation) isa NamedTuple
    mach_evals = SDM.machine_evaluations(evaluation)
    @test mach_evals isa NamedTuple{(:train, :test, :validation)}
    @test mach_evals.train isa NamedTuple{(keys(SDM.measures(evaluation)))}

    machine_aucs = SDM.machine_evaluations(evaluation).test.auc

    pr1 = SDM.predict(ensemble, backgrounddata)
    pr2 = SDM.predict(ensemble, backgrounddata; reducer = maximum)
    pr3 = SDM.predict(ensemble, backgrounddata; reducer = x -> sum(x .> 0.5), by_group = true)

    @test pr2 isa Vector
    @test collect(keys(pr1)) == SDM.machine_keys(ensemble)
    @test (keys(pr3)) == SDM.model_keys(ensemble)
    eltype(pr3) == Vector{Int64}

    @test_throws ArgumentError SDM.predict(ensemble, backgrounddata.a)
    @test_throws ArgumentError SDM.predict(ensemble, backgrounddata[(:a,)])
    @test_throws Exception SDM.predict(ensemble, backgrounddata; by_group = true)

    # explain
    expl = explain(ensemble; method = ShapleyValues(10; rng))
    varimp = variable_importance(expl)
    @test varimp.b > varimp.a
    @test varimp.c > varimp.a

    # plots
    interactive_evaluation(ensemble, thresholds = 0:0.001:1)
    interactive_response_curves(expl)
    boxplot(evaluation, :auc)
end

@testset "collinearity" begin
    # mock data with a collinearity problem
    data_with_collinearity = merge(backgrounddata, (; d = backgrounddata.a .+ rand(rng, n), e = backgrounddata.a .+ rand(rng, n), f = f = categorical(rand(Distributions.Binomial(3, 0.5), n))))

    rm_col_gvif = remove_collinear(data_with_collinearity; method = SDM.Gvif(; threshold = 2.), silent = false)
    rm_col_vif = remove_collinear(data_with_collinearity; method = SDM.Vif(; threshold = 2.), silent = true)
    rm_col_pearson = remove_collinear(data_with_collinearity; method = SDM.Pearson(; threshold = 0.65), silent = true)
    @test rm_col_gvif == (:b, :c, :d, :e, :f)
    @test rm_col_vif == (:b, :c, :d, :e, :f)
    @test rm_col_pearson == (:b, :c, :d, :e, :f)

    data_with_perfect_collinearity = (a = [1,2,3], b = [1,2,3])
    Test.@test_throws Exception remove_collinear(data_with_perfect_collinearity; method = SDM.Gvif(; threshold = 2., remove_perfectly_collinear = false), silent = true)
    @test remove_collinear(data_with_perfect_collinearity; method = SDM.Gvif(; threshold = 2.), silent = true) == (:a, )
    @test remove_collinear(data_with_perfect_collinearity; method = SDM.Pearson(; threshold = 0.65), silent = true) == (:a, )
end

