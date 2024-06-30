# Convert a BitArray to a CategoricalArray. Faster and type-stable version of `categorical`
BooleanCategorical{N} = CategoricalArrays.CategoricalArray{Bool, N, UInt8} where N
function boolean_categorical(A::BitArray{N}) where N 
    BooleanCategorical{N}(A, levels=[false, true], ordered=false)
end
boolean_categorical(A::AbstractVector{Bool}) = boolean_categorical(BitArray(A))


struct SDMdata{K}
    predictor::NamedTuple
    response::CategoricalArrays.CategoricalArray
    geometry::Union{Nothing, Vector}
    traintestpairs::MLJBase.TrainTestPairs
    resampler::Union{Nothing, MLJBase.ResamplingStrategy}

    function SDMdata(predictor::P, response, geometry, traintestpairs, resampler) where P<:NamedTuple{K} where K
        new{K}(predictor, response, geometry, traintestpairs, resampler)
    end
end

function Base.show(io::IO, mime::MIME"text/plain", data::SDMdata{K}) where K
    y = response(data)
    print("SDMdata object with ")
    printstyled(sum(y), bold = true)
    print(" presence points and ")
    printstyled(length(y) - sum(y), bold = true)
    print(" absence points. \n \n")

    print("Data is divided into $(nfolds(data)) folds. \n \n")

    printstyled("Predictor variables: \n", bold = true)
    Base.show(io, mime, MLJBase.schema(predictor(data)))

    if isnothing(geometry(data)) 
        print("Does not contain geometry data")
    else
        print("Also contains geometry data")
    end
end


_gettrainrows(d::SDMdata, i) = d.traintestpairs[i][1]
_gettestrows(d::SDMdata, i) = d.traintestpairs[i][2]
predictor(d::SDMdata) = d.predictor
predictorkeys(d::SDMdata{K}) where K = K 
response(d::SDMdata) = convert(AbstractArray{Bool}, d.response)
geometry(d::SDMdata) = d.geometry
traintestpairs(d::SDMdata) = d.traintestpairs
nfolds(d::SDMdata) = length(d.traintestpairs)

function _sdmdata(presences, absences, resampler, ::Nothing)
    predictorkeys = Tuple(Base.intersect(Tables.schema(presences).names, Tables.schema(absences).names))
    length(predictorkeys) > 0 || error("Presence and absence data have no common variable names - can't fit the ensemble.")
    _sdmdata(presences, absences, resampler, predictorkeys)
end

function _sdmdata(presences, absences, resampler, predictorkeys::NTuple{<:Any, <:Symbol})
    X, y = _predictor_response_from_presence_absence(presences, absences, predictorkeys)
    _sdmdata(X, y, resampler, predictorkeys)
end

# in case input is a table
function _sdmdata(X, response::BitVector, resampler, ::Nothing)
    columns = Tables.columns(X)
    Tables.rowcount(columns) == length(response) || error("Number of rows in predictors and response do not match")
    predictorkeys = Tables.columnnames(columns)
    _sdmdata(X, response, resampler, predictorkeys)
end

_sdmdata(X::Tables.ColumnTable{K}, y::BitVector, resampler, predictorkeys::NTuple{<:Any, <:Symbol}) where K =
    if K == predictorkeys
        _sdmdata(X, boolean_categorical(y), resampler)
    else
        _sdmdata(X[predictorkeys], boolean_categorical(y), resampler)
    end


function _sdmdata(
    X::Tables.ColumnTable, 
    y::BooleanCategorical, 
    resampler::MLJBase.ResamplingStrategy, 
)
    geometries = :geometry ∈ keys(X) ? Tables.getcolumn(X, :geometry) : nothing
    X = Base.structdiff(X, NamedTuple{(:geometry,)})
    traintestpairs = MLJBase.train_test_pairs(resampler, eachindex(y), X, y, geometries)
    SDMdata(X, y, geometries, traintestpairs, resampler)
end

function _sdmdata(
    X::Tables.ColumnTable, 
    y::BooleanCategorical, 
    resampler::MLJBase.TrainTestPairs, 
)
    SDMdata(X, y, geometries, traintestpairs, nothing)
end

cpu_backend(threaded) = threaded ? CPUThreads() : CPU1()
_map(::CPU1) = Base.map
_map(::CPUThreads) = ThreadsX.map


function _predictor_response_from_presence_absence(presences, absences, predictorkeys::NTuple{<:Any, <:Symbol})
    p_columns = Tables.columns(presences)
    a_columns = Tables.columns(absences) 
    n_presence = Tables.rowcount(p_columns)
    n_absence = Tables.rowcount(a_columns)

    # merge presence and absence data into one namedtuple of vectors
    X = NamedTuple{predictorkeys}([[a_columns[var]; p_columns[var]] for var in predictorkeys])
    y = [falses(n_absence); trues(n_presence)]
    return (X, y)
end