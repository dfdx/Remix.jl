###############################################################################
#                                    Grad                                     #
###############################################################################


getderiv(tape::Tape, v::Variable) = get(tape.c.derivs, V(v.id), nothing)
setderiv!(tape::Tape, x::Variable, dx::Any) = (
    tape.c.derivs[V(x.id)] = dx isa V ? V(tape, dx.id) : dx
)
hasderiv(tape::Tape, v::Variable) = getderiv(tape, v) !== nothing


function set_or_add_deriv!(tape::Tape, x::Variable, dx::Any)
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
            trg_v = push!(t.tape, copy(src_op))
            outs[src_v] = trg_v
        end
    end
end


function vjp_bwd!(t::Tracer, src::Tape)
    # backward pass
    seed = push!(t.tape, Constant(1))
    outs = t.tape.c.outs
    derivs = t.tape.c.derivs
    residuals = t.tape.c.residuals
    derivs[V(src.result.id)] = seed
    for src_op in reverse(src.ops)
        if src_op isa Call
            src_args = Umlaut.map_vars(v -> V(v.id), src_op.args)  # unbind from src
            trg_args = [get(outs, v, v) for v in src_args]
            src_v = V(src_op.id)
            res = residuals[src_v]
            g = derivs[src_v]
            dxs_t = trace!(t, (vjp_bwd, res, g, src_op.fn, trg_args...))
            @assert dxs_t.op.val isa Tuple
            global STATE = t, src, src_op, dxs_t
            dxs_op = t.tape[dxs_t]
            dxs = if dxs_op isa Call
                dxs_op.args
            elseif dxs_op isa Constant
                dxs_op.val
            else
                throw(AssertionError("Unexpected operation type of derivative: $dxs_op"))
            end
            for (x, dx) in zip(src_op.args, dxs)
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
    tape = remove_unused(tape)
    return val, tape[tape.result].val
end
