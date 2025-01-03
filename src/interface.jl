"""
    sdmdata(presences, absences; resampler, predictors)
    sdmdata(X, y::BitVector; resampler, predictors)

Construct an `SDMdata` object from species `presences` and `absences`. 
Alternatively, from a table with predictor variables `X` and a `BitVector` `y`, where `false` represents absence and `true` represents presence.

## Keywords
- `resampler`: The resampling strategy to be used. Should be a `MLJBase.ResamplingStrategy`, or a `Vector` of `Tuple`s with the form `(train, test)`. 
    Defaults to `NoResampling()`. If `resampler` is a `CV`, `shuffle` is internally set to `true`.
- `predictors`: a `Tuple` of `Symbols` with the names of the predictor values to be used. By default, all predictor variables in `X`,
   or all predictor variables in both `presences` and `absences` are used..

## Returns
An `SDMdata` object containing the data provided. This object can be used to construct an `SDMensemble`.

## Example
```
using Rasters, SpeciesDistributionModels
A = rand(10,10)
B = rand(10,10)
st = RasterStack((a=A, b=B), (X, Y); missingval=(a=missing,b=missing))

presence_points = [(1, 1), (2, 2), (3, 3), (4, 4)]
absence_points = [(5, 5), (6, 6), (7, 7), (8, 8)]

p = extract(st, presence_points)
a = extract(st, absence_points)

mydata = sdmdata(p, a; resampler = CV(nfolds = 2)) # 2-fold cross validation
mydata2 = sdmdata([p; a], [trues(4); falses(4)]; resampler = [([1,2],[5,6]), ([3,4], [7,8])]) # provide resampling rows
```
"""
function sdmdata(
    presences,
    absences;
    resampler = NoResampling(),#MLJBase.CV(; nfolds = 5, shuffle = true),
    predictors = nothing,
)
    _sdmdata(presences, absences, resampler, predictors)
end

"""
    sdm(data, models; [resampler], [predictors], [verbosity])

Construct an ensemble.

## Arguments
`data`: an `SDMdata` object
`models`: a `NamedTuple` with the models to be used in the ensemble.

## Keywords
- `models`: a `Vector` of the models to be used in the ensemble. All models must be MLJ-supported Classifiers. 
- For a full list of supported models, see https://alan-turing-institute.github.io/MLJ.jl/stable/model_browser/#Classification
- `predictors`: a `Vector` of `Symbols` with the names of the predictor values to be used. By default, all pdf
- `verbosity`: an `Int` value that regulates how much information is printed.
- `cache`: is passed to `MLJBase.machine`. Specify cache=false to prioritize memory management over speed.
- `scitype_check_level`: is passed to `MLJBase.machine`. Specify scitype_check_level=0 to disable scitype checking.

## Example
```julia
using SpeciesDistributionModels, Maxnet, MLJGLMInterface
mydata = sdmdata(presences, absences; resampler = CV(nfolds = 5))
models = (maxnet = MaxnetBinaryClassifier(), glm = LinearBinaryClassifier())
ensemble = sdm(mydata, models)
```
"""
function sdm(
    data, models;
    verbosity = 0,
    cache = true,
    scitype_check_level = 1,
    threaded = false
)
    _sdm(data, models, verbosity, cache, scitype_check_level, threaded)
end

"""
    evaluate(x; measures, train = true, test = true, [validation])

Evaluate `x`, which could be a SDMmachine, SDMgroup, or SDMensemble, 
by applying the measures provided to the data used to built an ensemble, 
and return an evaluation object.

## Keywords
- `measures` is a `NamedTuple` of measures. The keys are used to identify the measures.
This defaults to using auc, log_loss, and kappa.
For threshold-dependent measures, the highest score as well as the threshold at which the highest scores is reached are reported.
A list of measurse is available here: https://juliaai.github.io/StatisticalMeasures.jl/dev/auto_generated_list_of_measures/#aliases. However, note that not all measures are useful.
- `train`: a `Bool` that indicates whether to evaluate on the training data. Defaults to `true`
- `test`: a `Bool` that indicates whether to evaluate on the test data.Defaults to `true`
- `validation`: optionally, provide a separate validation dataset. 
Data should be provided as a `Tuple` with presences as the first field and absences as the second.
"""
function evaluate( # Define this as an extension of MLJBase.evaluate??
    x;
    measures = (; 
        StatisticalMeasures.accuracy,
        StatisticalMeasures.auc, 
        StatisticalMeasures.log_loss, 
        StatisticalMeasures.kappa
    ),
    train = true,
    test = true,
    validation::Tuple = ()
)
    test || train || isempty(validation) || error("No data to test. Either test or train must be true, or validation data must be provided")
    if !isempty(validation)
        X, y = _predictor_response_from_presence_absence(validation[1],validation[2], predictorkeys(data(x)))
        validation = (X, boolean_categorical(y))
    end

    _evaluate(x, measures, train, test)
end


"""
    explain(ensemble::SDMensemble; method, [data], [predictors])

Generate response curves for `ensemble`.

## Keywords
- `method` is the algorithm to use. See ShapleyValues
- `data` is the data to use to generate response curves, and defaults to the data used to train the ensemble
- `predictors`: which predictors to generate response curves for. Defaults to all variables in `data`.

"""
function explain(e::SDMensemble; method, data = data(e).predictor, predictors = keys(data))
    _explain(e, method, data, predictors)
end

"""
    predict(SDMobject, newdata; clamp = false, threaded = false)

Use an `SDMmachine`, or `SDMensemble` to predict habitat suitability for some data, optionally summarized for the entire ensemble, or for each `SDMgroup`.

`newdata` can be either a `RasterStack`, or a Tables.jl.compatible object. It must have all predictor variables used to train the models in its columns (or layers in case of a RasterStack).

## Keywords
- `clamp`: if `true`, the predictions are clamped to the interval seen during training of `SDMobject`. Defaults to `false`
- `threaded`: if `true`, run multithreaded. Defaults to `true`.
- `reducer`: Optionally provide a `Function` to summarize the output. The function should take an vector of values and return a single value. Typical examples are `Statistics.mean` or `Statistics.median`.
- `by_group` is set to `true`, the data is reduced for each `SDMgroup`, if it is set to `false` (the default), it reduced across the entire ensemble.

## Returns
If `newdata` is a `RasterStack`, the `predict` return a `Raster`; otherwise, return a `DimArray`.  
Habitat suitability represented by a floating-point number between 0 and 1.
"""
function predict(m::SDMmachine, d; clamp = false)
    _reformat_and_predict(m, d, clamp)
end
function predict(e::SDMensemble, d; clamp = false, threaded = false)
    _reformat_and_predict(e, d, clamp, threaded)
end



