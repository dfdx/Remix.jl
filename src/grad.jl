###############################################################################
#                                  Context                                    #
###############################################################################


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


###############################################################################
#                                    Grad                                     #
###############################################################################


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
