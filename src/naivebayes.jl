module NaiveBayes

import ..Common
import ..Model
import ..Data

import ..Model: Probability, Counting, getlogp, estPMin, estExtra

import Distributions: Normal, pdf, logpdf

# For NB we transform probabilities into log probabilities
function getlogp(M::Probability, label::Int64, v::Data.DiscValue)
  M.disc.conditional[ label, v.index ]
end

function getlogp(M::Probability, label::Int64, v::Data.DiscValue)
  M.disc.conditional[ label, v.index ]
end


typealias SparseMat SparseMatrixCSC{Float64,Int64}

type Settings
  ignorePriors::Bool

  function Settings(params::Common.Params)
    new( get(params, "ignorePriors", true) )
  end
end

type NB
  shape::Data.Shape
  counts::Model.Counting
  settings::Settings

  function NB(shape::Data.Shape, params::Common.Params)
    new(shape, Model.Counting(shape), Settings(params))
  end
end

type NBPredModel
  shape::Data.Shape
  p::Model.Probability
  pZero::Matrix{Float64}
  pLabel::Vector{Float64}


  function NBPredModel(nb::NB)
    NBPredModel(nb.counts, nb.settings.ignorePriors)
  end

  function NBPredModel(counts::Model.Counting, ignorePriors::Bool)
    shape = counts.shape
    p = Model.Probability(counts)
    p.disc.conditional = splog( p.disc.conditional )
    p.label = map(log, p.label)

    pMin = estPMin(counts.disc, estExtra(counts.disc))

    pZ = zeros(shape.labels, shape.discFeatures + shape.realFeatures)
    pB = ignorePriors ? zeros(shape.labels) : copy(p.label)

    for feature=1:(shape.discFeatures + shape.realFeatures)
      if shape.labelAttr != feature
        tID = shape.index[feature]
        if shape.isReal[feature]

          for label=1:shape.labels
            pZ[label,tID] = logpdf(p.normal.conditional[label,tID], 0)
            pB[ label ] += pZ[label,tID]
          end

        else

          frng = shape.offsets[tID]:(shape.offsets[tID+1]-1)

          for label=1:shape.labels
            z = 1
            for j=frng
              z -= p.disc.conditional[label, j]
            end
            z = log(z)

            pZ[label, feature] = z
            pB[ label ] += z
          end

        end
      end

    end

    new(shape, p, pZ, pB)
  end
end

function splog(m::SparseMat)
  lp = spzeros( size(m,1), size(m,2) )

  I,J,V = findnz(m)
  for (i,j,v) in zip(I,J,V)
    lp[i,j] = log(v)
  end

  return lp
end

function train(model::NB, data::Data.Dataset)
  for row in Data.eachrow(data)
    Model.push!(row, model.counts)
  end

  return model
end

function label(model::NB, params::Common.Params, stream::Task)

  const produceRanks = get(params, "ranks", false)

  predModel = NBPredModel(model)
  classRange = 1:model.shape.labels

  if produceRanks
    for row in stream
      produce( (label(predModel, row, produceRanks), row.labels) )
    end
  else
    for row in stream
      produce( (label(predModel, row, produceRanks), row.labels) )
    end
  end

  return model
end

function label(model::NBPredModel, row::Data.Row, produceRanks::Bool)
  # Initialize the probabilities
  p = copy(model.pLabel)
  labelRng = 1:length(p)

  for v in row.values
    for c=labelRng
      p[c] += getlogp(model.p, c, v) - model.pZero[c, v.index]
    end
  end


  if produceRanks
    return [ Common.Ranking(i, p[i]) for i=labelRng ]
  else
    return indmax(p)
  end

end

export train, label, NB, NBPredModel
end # module
