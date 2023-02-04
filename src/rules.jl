@constcall Core.apply_type(u::UnionAll, T)
@constcall tuple(x::Integer)
@constcall NamedTuple(::Tuple{<:Integer})
@constcall Core.kwfunc(f)


vjp_fwd(::typeof(*), x::Number, y::Number) = x * y, ()
vjp_bwd(res, g, ::typeof(*), x::Number, y::Number) = (g * y, g * x)

vjp_fwd(::typeof(*), x::AbstractMatrix, y::AbstractMatrix) = x * y, (x, y)
vjp_bwd(res, g, ::typeof(*), x::AbstractMatrix, y::AbstractMatrix) = (g * y', x' * g)


vjp_fwd(::typeof(+), x::Number, y::Number) = (x + y, ())
vjp_bwd(res, g, ::typeof(+), x::Number, y::Number) = (g, g)

# TODO: rewrite to consist only of primitives
# _unsum(x, dy, dims) = broadcast(last∘tuple, x, dy)
# _unsum(x, dy, ::Colon) = broadcast(last∘tuple, x, Ref(dy))


function _unsum(x, dy, dims)
#     sz = @constexpr [size(x, d) for d in dims]
end


vjp_fwd(::typeof(Core.kwfunc(sum)), kw, ::typeof(sum), x::AbstractArray) = sum(x, dims=kw.dims), (x,)
vjp_bwd(res, g, ::typeof(Core.kwfunc(sum)), kw, ::typeof(sum), x::AbstractArray) = _unsum(x, g, dims)
