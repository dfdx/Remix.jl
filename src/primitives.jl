tile(A::AbstractArray, reps::Tuple{<:Integer}) = repeat(A, outer=reps)