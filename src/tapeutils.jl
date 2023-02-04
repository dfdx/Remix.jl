function dependencies!(tape::Tape, y_id::Int, result::Set{Int})
    push!(result, y_id)
    y = V(tape, y_id)
    # y_fargs = is_kwfunc(y._op.fn) ? tape[y].args[3:end] : tape[y].args
    y_fargs = tape[y].args
    for x in y_fargs
        if x isa V && !in(x.id, result) && tape[x] isa Call
            dependencies!(tape, x.id, result)
        end
    end
end

"""
Collect variables that we need to step through during the reverse pass.
The returned vector is already deduplicated and reverse-sorted
"""
function dependencies(tape::Tape)
    @assert(tape[tape.result] isa Call, "The tape's result is expected to be a Call, " *
            "but instead $(typeof(tape[tape.result])) was encountered")
    result = Set{Int}()
    dependencies!(tape, tape.result.id, result)
    ids = sort(collect(result), rev=true)
    return Set(ids)
end


"""
    copy(op::Umlaut.AbstractOp)

Make a shallow copy of an op. Shallow copy is a lightway object that references
original values, reducing memory pressure. At the same time, a user can modify or even
corrupt the original data via the copied operation.
"""
function Base.copy(op::Umlaut.AbstractOp)
    T = typeof(op)
    fld_names = fieldnames(T)
    fld_vals = [getfield(op, fld) for fld in fld_names]
    return T(fld_vals...)
end


"""
    remove_unused(tape::Tape)

Create a new tape containing only operations from the original tape that are used
to produce the result
"""
function remove_unused(tape::Tape)
    deps = dependencies(tape)
    new_tape = Tape(tape.c)
    # TODO: map old to new vars
    old2new = Dict{Int, Int}()
    for op in tape
        if op isa Input || (op isa Constant && op.id in deps)
            new_v = push!(new_tape, copy(op))
            old2new[op.id] = new_v.id
        elseif op isa Call || op.id in deps
            new_v_fargs = [
                v isa V ? V(new_tape, old2new[v.id]) : v for v in (op.fn, op.args...)
            ]
            new_v_f, new_v_args... = new_v_fargs
            new_op = mkcall(new_v_f, new_v_args...; line=op.line, val=op.val)
            new_v = push!(new_tape, new_op)
            old2new[op.id] = new_v.id
        end
    end
    new_result_id = old2new[tape.result.id]
    new_tape.result = V(new_tape, new_result_id)
    return new_tape
end