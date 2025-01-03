function _reformat_and_predict(e, d, clamp, args...)
    Tables.istable(d) || throw(ArgumentError("data is a $(typeof(d)), wich is not a Tables.jl-compatible table"))
    data = _select_features(sdmdata(e), d)
    output = _allocate_prediction(data, DD.dims(d), DD.dims(e))
    _predict!(output, data, e, args...)
end

function _select_features(d::SDMdata, x)
    cols = Tables.Columns(x)
    colnames = Tables.columnnames(cols)
    for key in predictorkeys(d)
        key in colnames || throw(ArgumentError("data is missing predictor variable $key"))
    end
    return NamedTuple(K => Tables.getcolumn(cols, K) for K in predictorkeys(d))
end
function _select_features(d::SDMdata, rs::Rasters.AbstractRasterStack)
    layernames = Rasters.name(rs)
    for key in predictorkeys(d)
        key in layernames || throw(ArgumentError("data is missing predictor variable $key"))
    end
    return rs[predictorkeys(d)]
end

_allocate_prediction(data, ::Nothing, ::Nothing) = _allocate_prediction(data, (Dim{:row}(1:Tables.rowcount(data)),))
_allocate_prediction(data, ::Nothing, ensdims) = _allocate_prediction(data, (Dim{:row}(1:Tables.rowcount(data)), ensdims...))
_allocate_prediction(data, dims, ensdims::Nothing) = _allocate_prediction(data, dims)
_allocate_prediction(data, dims, ensdims) = _allocate_prediction(data, (dims..., ensdims...))
_allocate_prediction(data, dims::DD.DimTuple) = zeros(dims)
function _allocate_prediction(st::RA.AbstractRasterStack, dims::DD.DimTuple)
    data = fill!(RA.Raster{Union{Missing, Float64}}(undef, dims), missing) 
    DD.rebuild(first(DD.layers(st)); dims, data, name = DD.NoName())
end

function _predict!(output::RA.AbstractRaster, data::RA.AbstractRasterStack, e::SDMensemble, threaded::Bool)
    bm = Rasters.boolmask(data)
    outputslices = @views [output[I][bm] for I in DD.DimIndices(DD.dims(e))]
    _predict!(outputslices, data[bm], e, threaded)
    return output
end
function _predict!(output::RA.AbstractRaster, data::RA.AbstractRasterStack, e::SDMmachine)
    bm = Rasters.boolmask(data)
    @views _predict!(output[bm], data[bm], e)
    return output
end
function _predict!(output::DimArray, data, e::SDMensemble, threaded::Bool)
    @maybe_threads threaded for I in DD.DimIndices(e)
        _predict!(view(output, I), data, e[I])
    end
    return output
end
function _predict!(output::AbstractArray, data, m::SDMmachine)
    prediction = MLJBase.predict(m.machine, data)
    vec(output) .= MLJBase.pdf.(prediction, true)
    return output
end
# view(output, I) can result in a 0-dimension array
_predict!(output::DimArray{<:AbstractArray, 0}, data, m::SDMmachine) = _predict!(first(output), data, m)

#=
_


# Reformat data so that it can be used in predict. Different models use different data types
function _reformat_data(m::SDMmachine, d, clamp::Bool)
    traindata = predictor(data(m))
    newdata = Tables.columntable(d)[keys(traindata)]
    if clamp
        for k in keys(traindata)
            if !(MLJBase.scitype(traindata[k]) <: AbstractVector{<:MLJBase.Finite}) # if data is categorical, don't clamp
                newdata[k] .= Base.clamp.(newdata[k], Base.extrema(traindata[k])...)
            end
        end
    end

    return MLJBase.reformat(MLJBase.last_model(m.machine), newdata)[1]
end
#### _predict methods ####
# _predict uses already-reformatted data.
# _reformat_and_predict methods first reformat and then call _predict

# Machine-level _predict method. All other _predict methods eventually call this
function _predict(m::SDMmachine, data, dims)
    # predict
    mach = m.machine
    prediction = MLJBase.predict(MLJBase.last_model(mach), mach.fitresult, data)
    prediction = MLJBase.get!(prediction, :predict, mach)
    # convert to Floats
    return DimVector(MLJBase.pdf.(prediction, true), dims)
end

function _reformat_and_predict(m::SDMmachine, data, clamp)
    _predict(m, _reformat_data(m, data, clamp))
end

## Group level _predict methods
function _predict(g::SDMgroup, data, ::Nothing, resource::AbstractCPU)
    pr = _map(resource)(m -> _predict(m, data), g)
    return NamedTuple{Tuple(machine_keys(g))}(pr)
end

function _predict(g::SDMgroup, data, reducer::Function, resource::AbstractCPU) 
    pr = _map(resource)(m -> _predict(m, data), g)
    return  map((d...) -> reducer(d), pr...)
end

function _reformat_and_predict(g::SDMgroup, data, clamp, reducer, resource::AbstractCPU)
    _predict(g, _reformat_data(first(g), data, clamp), reducer, resource)
end

# ensemble-level methods
# For ensemble, there are no _predict methods. Datas has to be reformatted for each group
function _reformat_and_predict(e::SDMensemble, data, clamp::Bool, ::Nothing, ::Bool, resource::AbstractCPU)
    pr = _map(resource)(
        g -> _reformat_and_predict(g, data, clamp, nothing, resource), 
        e
    )
    merge(pr...)
end

function _reformat_and_predict(e::SDMensemble, data, clamp::Bool, reducer::Function, by_group::Bool, resource::AbstractCPU)
    if by_group
        # pass the reducer to each group, then combine into a namedtuple
        group_pr = _map(resource)(g -> _reformat_and_predict(g, data, clamp::Bool, reducer, resource), e)
        NamedTuple{model_keys(e)}(group_pr)
    else
        # predict without reducing, then apply the reducer
        pr = _map(resource)(g -> _reformat_and_predict(g, data, clamp::Bool, nothing, resource), e)
        pr_m = merge(pr...)
        map((d...) -> reducer(d), pr_m...)
    end
end

# Dispatch on RasterStacks
_reformat_and_predict(e::SDMensemble, rs::Rasters.AbstractRasterStack, clamp::Bool, reducer::Function, by_group::Bool, resource::AbstractCPU) = 
    _reformat_and_predict_raster(e, rs, clamp, reducer, by_group, resource)
_reformat_and_predict(e::SDMensemble, rs::Rasters.AbstractRasterStack, clamp::Bool, reducer::Nothing, by_group::Bool, resource::AbstractCPU) = 
    _reformat_and_predict_raster(e, rs, clamp, reducer, by_group, resource)
_reformat_and_predict(g::SDMgroup, rs::Rasters.AbstractRasterStack, clamp::Bool, reducer::Union{<:Function, <:Nothing}, resource::AbstractCPU) = 
    _reformat_and_predict_raster(g, rs, clamp, reducer, resource)
_reformat_and_predict(m::SDMmachine, rs::Rasters.AbstractRasterStack, clamp::Bool) =
    _reformat_and_predict_raster(m, rs, clamp)

function _reformat_and_predict_raster(s::Union{<:SDMensemble, SDMgroup, SDMmachine}, rs::Rasters.AbstractRasterStack, args...)
    rs_preds = rs[predictorkeys(data(s))]
    missing_mask = Rasters.boolmask(rs_preds)
    d = rs_preds[missing_mask]
    if any(Rasters.maplayers(x -> Missing <: eltype(x), rs_preds))
        # to get rid of Union{Missing, Float64} etc.
        layertypes = map(x -> Base.nonmissingtype(eltype(x)), (Rasters.layers(rs_preds)))
        nttype = NamedTuple{keys(layertypes), Tuple{values(layertypes)...}}
        d = nttype.(d)
    end
    pr =  _reformat_and_predict(s, d, args...)
    return _build_raster(missing_mask, pr)
end

# Build Raster with the models/machines in Band dimension
function _build_raster(missing_mask::Rasters.AbstractRaster, pr::NamedTuple)
    r_dims = (Rasters.dims(missing_mask)..., Band(collect(keys(pr))))
    T = eltype(first(pr)) # usually Float64, but could be something else depending on the reducer
    output = Raster(Array{Union{Missing, T}}(missing, size(r_dims)); dims = r_dims)
    for k in keys(pr)
        @views output[Band = Rasters.At(k)][missing_mask] .= pr[k]
    end
    return output
end

# Build Raster with no additional layer
function _build_raster(missing_mask::Rasters.AbstractRaster, pr::Vector)
    r_dims = Rasters.dims(missing_mask)
    T = eltype(pr) # usually Float64, but could be something else depending on the reducer
    output = Raster(Array{Union{Missing, T}}(missing, size(r_dims)), dims = r_dims)
    output[missing_mask] .= pr
    return output
end
=#