module Common

    using JSON

    typealias Params Dict{String,Any}

    type Ranking
        class::Int64
        value::Float64
    end


export Ranking, Params
end
