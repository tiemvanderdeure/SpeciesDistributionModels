"""
    thin([rng], x, cutoff; distance = Haversine(), [geometrycolumn])

    Thin spatial data by removing points that are closer than `cutoff` distance
    to the nearest other point in the dataset.

    ## Arguments
    - `rng`: a random number generator. The default is `Random.GLOBAL_RNG()`.
    - `x`: an `AbstractVector` that iterates points, or a table with a `:geometry` column.
    - `cutoff`: the distance threshold in units of `distance`.
    ## Keywords
    - `distance`: the distance metric used to calculate distances between points. The default
    is `Haversine()`, which uses the Haversine formula to calculate the distance between coordinates in meter units.
    - `geometrycolumn`: the name of the column in the table that contains the points, if `x` is a table. Usually defaults to `:geometry`.

    ## Example
    ```jldoctest
    using SpeciesDistributionModels, Distances
    # a vector that iteratores points
    geometries = [(0,0), (1,0), (0,0.000001)]
    # thin to 1000 meters
    thin(geometries, 1000)
    # thin to 1 degree
    thin(Xoshiro(123), geometries, 1; distance = Euclidean())

    # output
    2-element Vector{Tuple{Int64, Real}}:
    (0, 0)
    (1, 0)
    ```
"""
thin(x, cutoff; kw...) = thin(Random.GLOBAL_RNG, x, cutoff; kw...)

function thin(rng::Random.AbstractRNG, x, cutoff; distance = Distances.Haversine(), geometrycolumn = nothing) # = first(GI.geometrycolumn(x))
    if !(x isa AbstractVector{<:GI.NamedTuplePoint}) && Tables.istable(x)
        geomcol = isnothing(geometrycolumn) ? first(GI.geometrycolumns(data)) : geometrycolumn
        geoms = Tables.getcolumn(Tables.Columns(x), geomcol)
        indices = _thin(rng, geoms, cutoff, distance)
        return Tables.subset(x, indices)
    elseif x isa AbstractVector
        geoms = unique(x)
        indices = _thin(rng, geoms, cutoff, distance)
        return geoms[indices]
    else
        throw(ArgumentError("x must be an AbstractVector or a Table with a :$geometrycolumn column."))
    end
end

function _thin(rng::Random.AbstractRNG, geoms::AbstractVector, cutoff::Real, distance::Distances.Metric)
    dist_matrix = Distances.pairwise(distance, geoms)

    dist_mask = dist_matrix .< cutoff
    dist_sum = vec(sum(dist_mask; dims = 1))
    s = size(dist_sum, 1)

    indices = collect(1:s)

    for i in 1:s
        m = maximum(dist_sum)
        if m == 1
            break
        else
            drop = rand(rng, findall(dist_sum .== m))
            dist_sum .-= @view dist_mask[drop, :]
            drop_indices = 1:s .!= drop
            dist_mask = @view dist_mask[drop_indices, drop_indices]
            dist_sum = @view dist_sum[drop_indices]
            s -= 1
            indices = @view indices[drop_indices]
        end
    end
    return indices
end
