using Umlaut
import Umlaut: V, Call, Tracer, trace!, inputs!, inputs


func(x, y) = sin(x) * y

function vjp_fwd(func::typeof(func), x, y)
    return func(x, y), (cos(x), sin(x), y)
end

function vjp_bwd(func::typeof(func), res, g)
    cos_x, sin_x, y = res
    return (cos_x * g * y, sin_x * g)
end


struct RemixCtx
    # in the mappings below keys are unboud variables in source tape
    # values are bound variables in target tape
    outs::Dict{V, V}        # V(src_v.id) => vjp_fwd(...)[1]
    residuals::Dict{V, V}   # V(src_v.id) => vjp_fwd(...)[2]
    derivs::Dict{V, V}      # V(src_v.id) => g
end

RemixCtx() = RemixCtx(Dict(), Dict(), Dict())
Base.show(io::IO, ctx::RemixCtx) =
    print(io, "RemixCtx($(length(ctx.outs)) vars, $(length(ctx.derivs)) derivs)")


function Umlaut.isprimitive(::RemixCtx, f, args...)
    @nospecialize
    F = Core.Typeof(f)
    Args = Core.Typeof.(args)
    return Core.Compiler.return_type(vjp_fwd, Tuple{F, Args...}) !== Nothing
end


vjp_fwd(::typeof(*), x::Number, y::Number) = x * y, (x, y)
vjp_bwd(::typeof(*), res, g) = (g * res[2], g * res[1])

vjp_fwd(::typeof(+), x::Number, y::Number) = (x + y, ())
vjp_bwd(::typeof(+), res, g) = (g, g)



getderiv(tape::Tape, v::Variable) = get(tape.c.derivs, V(v.id), nothing)
setderiv!(tape::Tape, x::Variable, dx::Variable) = (
    tape.c.derivs[V(x.id)] = V(tape, dx.id)
)
hasderiv(tape::Tape, v::Variable) = getderiv(tape, v) !== nothing


function set_or_add_deriv!(tape::Tape, x::Variable, dx::Variable)
    if !hasderiv(tape, x)
        setderiv!(tape, x, dx)
    else
        old_dx = getderiv(tape, x)
        new_dx = push!(tape, mkcall(+, dx, old_dx; line="updated deriv for $x"))
        # if tape[dx].val isa Tangent || tape[old_dx].val isa Tangent
        #     new_dx = push!(tape, mkcall(+, dx, old_dx; line="updated deriv for $x"))
        # else
        #     new_dx = push!(tape, mkcall(broadcast, +, dx, old_dx; line="updated deriv for $x"))
        # end
        setderiv!(tape, x, new_dx)
    end
end


# function todo_list!(tape::Tape{RemixCtx}, y_id::Int, result::Set{Int})
#     push!(result, y_id)
#     y = V(tape, y_id)
#     # since `y = getfield(rr, 2)`, we use arguments of the original rrule instead
#     y_fargs = is_kwfunc(y._op.fn) ? tape[y].args[3:end] : tape[y].args
#     for x in y_fargs
#         if x isa V && !in(x.id, result) && tape[x] isa Call
#             todo_list!(tape, x.id, result)
#         end
#     end
# end

# """
# Collect variables that we need to step through during the reverse pass.
# The returned vector is already deduplicated and reverse-sorted
# """
# function todo_list(tape::Tape{GradCtx})
#     @assert(tape[tape.result] isa Call, "The tape's result is expected to be a Call, " *
#             "but instead $(typeof(tape[tape.result])) was encountered")
#     result = Set{Int}()
#     todo_list!(tape, tape.result.id, result)
#     ids = sort(collect(result), rev=true)
#     return [V(tape, id) for id in ids]
# end


# call_values(op::Call) = Umlaut.var_values([op.fn, op.args...])


function vjp_fwd!(t::Tracer, src::Tape)
    outs = t.tape.c.outs
    residuals = t.tape.c.residuals
    for src_op in src
        src_v = V(src_op.id)
        if src_op isa Call
            src_args = Umlaut.map_vars(v -> V(v.id), src_op.args)  # unbind from src
            trg_args = [get(outs, v, v) for v in src_args]
            trg_v = trace!(t, (vjp_fwd, src_op.fn, trg_args...))
            out, res = t.tape[trg_v].args
            outs[src_v] = out
            residuals[src_v] = res
        else
            trg_v = push!(t.tape, src_op)  # TODO: copy op
            outs[src_v] = trg_v
        end
    end
end


function vjp_bwd!(t::Tracer, src::Tape)
    # backward pass
    seed = push!(t.tape, Constant(1))
    derivs = t.tape.c.derivs
    residuals = t.tape.c.residuals
    derivs[V(src.result.id)] = seed
    for src_op in reverse(src.ops)
        if src_op isa Call
            src_v = V(src_op.id)
            res = residuals[src_v]
            g = derivs[src_v]
            dxs_t = trace!(t, (vjp_bwd, src_op.fn, res, g))
            @assert dxs_t.op.val isa Tuple
            for (x, dx) in zip(src_op.args, dxs_t.op.args)
                if x isa V
                    set_or_add_deriv!(t.tape, x, dx)
                end
            end
        end
    end
end


function grad(src::Tape)
    t = Tracer(Tape(RemixCtx()))
    vjp_fwd!(t, src)
    vjp_bwd!(t, src)
    return t.tape
end


function value_and_grad(f, args...)
    # TODO: add caching
    val, src = trace(f, args...; ctx=RemixCtx())
    tape = grad(src)
    gt = mkcall(tuple, [getderiv(tape, v) for v in inputs(src)[2:end]]...)
    tape.result = push!(tape, gt)
    return val, tape[tape.result].val
end



function main()
    f = (x, y) -> (x * y + 1) + x
    args = (2.0, 3.0)
    value_and_grad(f, args...)
end


# 1. trace function
# 2. replace f(args...) with subgraph from vjp_fwd(f, args...)
# 3. add vjp_bwd(res, g), mapping outputs from forward pass
# 4. utils to test vjps