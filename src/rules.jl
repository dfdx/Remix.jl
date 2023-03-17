@constcall Core.apply_type(u::UnionAll, T)
@constcall tuple(x::Integer)
@constcall NamedTuple(::Tuple{<:Integer})
@constcall Core.kwfunc(f)
@constcall findall(fn, x)


vjp_fwd(::typeof(*), x::Number, y::Number) = x * y, ()
vjp_bwd(res, g, ::typeof(*), x::Number, y::Number) = (g * y, g * x)

vjp_fwd(::typeof(*), x::AbstractMatrix, y::AbstractMatrix) = x * y, (x, y)
vjp_bwd(res, g, ::typeof(*), x::AbstractMatrix, y::AbstractMatrix) = (g * y', x' * g)


vjp_fwd(::typeof(+), x::Number, y::Number) = (x + y, ())
vjp_bwd(res, g, ::typeof(+), x::Number, y::Number) = (g, g)


vjp_fwd(::typeof(tile), x::AbstractArray, reps) = tile(x, reps), ()
vjp_bwd(res, g, ::typeof(tile), x::AbstractArray, reps) = (sum(g; dims=findall(x -> x != 1, reps)), nothing)

# TODO: rewrite to consist only of primitives
# _unsum(x, dy, dims) = broadcast(last∘tuple, x, dy)
# _unsum(x, dy, ::Colon) = broadcast(last∘tuple, x, Ref(dy))


"""
    reps_from_dims(x::AbstractArray, dims::Tuple{<:Integer})

If array `x` was reduced along `dims`, return number of repetitions needed
to restore the full size
"""
function reps_from_dims(x::AbstractArray, dims::Tuple{<:Integer})
    return tuple([d in dims ? size(x, d) : 1 for d in 1:ndims(x)]...)
end

@constcall reps_from_dims(x::AbstractArray, dims::Tuple{<:Integer})

function _unsum(x, dy, dims)
    reps = reps_from_dims(x, dims)
    return tile(dy, reps)    # or dy divided by number of elements?
end


vjp_fwd(::typeof(Core.kwfunc(sum)), kw, ::typeof(sum), x::AbstractArray) = sum(x, dims=kw.dims), (x,)
vjp_bwd(res, g, ::typeof(Core.kwfunc(sum)), kw, ::typeof(sum), x::AbstractArray) = _unsum(x, g, dims)
