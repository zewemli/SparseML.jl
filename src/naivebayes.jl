module NaiveBayes

import ..Common
import ..Model
import ..Data

typealias SparseMat SparseMatrixCSC{Float64,Int64}

type Settings
    ignorePriors::Bool

    function Settings(params::Common.Params)
        new( get(params, "ignorePriors", false) )
    end
end

type NB
    shape::Data.Shape
    counts::Model.Counting
    settings::Settings

    NB(shape::Data.Shape, params::Common.Params) =
        new(shape, Model.Counting(shape), Settings(params))
end

type NBPredModel
    pMin::Float64
    base::Vector{Float64}
    deltas::SparseMat

    rowProb::Vector{Float64}

    function NBPredModel(model::NB)
        NBPredModel(model.counts, model.settings.ignorePriors)
    end

    function NBPredModel(counts::Model.Counting, ignorePriors::Bool)
        NBPredModel( Model.Probability(counts), ignorePriors)
    end

    function NBPredModel(model::Model.Probability, ignorePriors::Bool)
        nclasses = size(model.prob,1)
        predModel = new(log(model.pMin), zeros(nclasses), spzeros(0,0), zeros( nclasses ))
        update(predModel, model, ignorePriors)
    end

end

function adjustProb!(probs::Vector{Float64}, features::Vector{Int64}, m::NBPredModel)
    for c=1:size(m.deltas,1)
        for j in features
            cj = m.deltas[c,j]
            probs[c] += (cj == 0) ? m.pMin : cj
        end
    end
    return probs
end

function update(pred::NBPredModel, model::Model.Probability, ignorePriors::Bool)
  deltas = spzeros( size(model.prob,1), size(model.prob,2) )
  pEmpty = zeros( size(model.pclass) )

  I,J,V = findnz(model.prob)
  for (c, f, p_f_given_c) in zip(I,J,V)
      p_neg = log(1 - p_f_given_c)
      pEmpty[c] += p_neg
      deltas[c,f] = log(p_f_given_c) - p_neg
  end

  if !ignorePriors
      pEmpty = .+(pEmpty, map(log, model.pclass))
  end

  pred.base = pEmpty
  pred.deltas = deltas

  return pred
end

function train(model::NB, data::Data.Dataset)
    for row in Data.eachrow(data, true)
        Model.countRow(row, model.counts)
    end

    return model
end

function label(model::NB, params::Common.Params, stream::Task)

    const produceRanks = get(params, "ranks", false)

    predModel = NBPredModel(model)
    classRange = 1:model.shape.classes

    for row in stream
        produce( (label(predModel, row, produceRanks), row.labels) )
    end

    return model
end

function label(model::NBPredModel, row::Data.Row, produceRanks::Bool)
    # Initialize the probabilities
    model.rowProb[1:end] = model.base
    features = [ Model.project(V, row.nbins) for V in row.values ]

    # Adjust the probabilities given the observed features
    adjustProb!(model.rowProb, features, model)

    if produceRanks
        return map((i) -> Common.Ranking(i, model.rowProb[i]), 1:length(model.rowProb))
    else
        return indmax(model.rowProb)
    end
end

export train, label, update, getProb, NB, NBPredModel
end # module
