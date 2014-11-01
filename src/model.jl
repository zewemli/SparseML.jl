module Model

import ..Data
import Distributions: Normal, pdf, logpdf

typealias SparseMat SparseMatrixCSC{Float64,Int64}

## Discrete Models
# Provide a model for counting overlaps and intersections
# this is the model which should be saved to disk

type DiscCount
  shape::Data.Shape
  conditional::SparseMat
  feature::Vector{Int64}
  n::Int64

  function DiscCount(shape::Data.Shape)
    new(shape,
        spzeros(shape.labels, shape.discWidth),
        zeros(shape.discWidth),
        0)
  end

  DiscCount() = DiscCount(Data.Shape())
end

type DiscProb
  shape::Data.Shape
  conditional::SparseMat
  feature::Vector{Float64}
  pMin::Float64

  function DiscProb(m::DiscCount, n::Int64, pLabel::Vector{Float64})
    m.n = n
    prob = new( m.shape,
               zeros(size(m.conditional)),
               zeros(size(m.feature)),
               estPMin(m, estExtra(m)) )

    extraN = estExtra( m )
    assert(extraN >= 0)

    labelCnt = sum(m.conditional, 2)
    pLabel = ./(labelCnt, sum(labelCnt))

    if m.n > 0
      prob.feature = ./(m.feature, m.n)

      I,J,V = findnz(m.conditional)
      for (c, f, f_and_c) in zip(I,J,V)
        cf_val = f_and_c / (labelCnt[c] + extraN*pLabel[c])
        prob.conditional[ c,f ] = isfinite(cf_val) ? cf_val : 0
      end
    end

    return prob
  end

end

## Normal Models
type GaussianStream
  n::Int64
  m::Float64
  s::Float64
  GaussianStream() = new(0,0.,0.)
end

type NormalCount
  shape::Data.Shape
  conditional::Matrix{GaussianStream}
  feature::Vector{GaussianStream}

  function NormalCount(shape::Data.Shape)
    new(shape,
        [ GaussianStream() for j=1:shape.labels, i=1:shape.realFeatures ],
        [ GaussianStream() for i=1:shape.realFeatures ])
  end
end

type NormalProb
  shape::Data.Shape
  conditional::Matrix{Normal}
  feature::Vector{Normal}

  function NormalProb(counts::NormalCount, pLabel::Vector{Float64})
    new(counts.shape,
        [ Normal(counts.conditional[i,j])
         for i=1:size(counts.conditional,1), j=1:size(counts.conditional,2) ],
        map(Normal, counts.feature))
  end
end

type Counting
  shape::Data.Shape
  disc::DiscCount
  normal::NormalCount
  label::Vector{Int64}
  n::Int64

  Counting(shape::Data.Shape) = new(shape,
                                    DiscCount(shape),
                                    NormalCount(shape),
                                    zeros(shape.labels), 0 )
end

type Probability
  shape::Data.Shape
  disc::DiscProb
  normal::NormalProb
  label::Vector{Float64}

  function Probability(counts::Counting)
    new(counts.shape,
        DiscProb(counts.disc, counts.n, ./(counts.label, counts.n)),
        NormalProb(counts.normal, ./(counts.label, counts.n)),
        ./(counts.label, counts.n) )
  end
end

function +(a::GaussianStream, b::GaussianStream)
  t = a.n + b.n
  pa = a.n / t
  pb = b.n / t
  GaussianStream(t, pa*a.m + pb*b.m, pa*a.s + pb*b.s)
end

# Get probability
function getp(M::Probability, v::Data.RealValue)
  pdf(M.normal.feature[ v.index ], v.value)
end

function getp(M::Probability, label::Int64, v::Data.RealValue)
  pdf(M.normal.conditional[ label, v.index ], v.value)
end

function getlogp(M::Probability, v::Data.RealValue)
  logpdf(M.normal.feature[ v.index ], v.value)
end

function getlogp(M::Probability, label::Int64, v::Data.RealValue)
  logpdf(M.normal.conditional[ label, v.index ], v.value)
end

function getp(M::Probability, v::Data.DiscValue)
  M.disc.feature[ v.value ]
end

function getp(M::Probability, label::Int64, v::Data.DiscValue)
  M.disc.conditional[ label, v.value ]
end

function getp(M::Probability, label::Int64, v::Data.DiscValue)
  M.disc.conditional[ label, v.value ]
end


function merge!(a::DiscCount,b::DiscCount)
  a.conditional += b.conditional
  a.label += b.label
  a.feature += b.feature
  a.n += b.n
  return a
end

function push!(s::Counting, label::Int64, v::Data.RealValue)
  push!(s.normal.conditional[label, v.index], v.value)
  push!(s.normal.feature[v.index], v.value)
end

function push!(s::Counting, label::Int64, v::Data.DiscValue)
  s.disc.conditional[ label, v.value ] += 1
  s.disc.feature[ v.value ] += 1
end

function push!(s::GaussianStream, x::Float64)
  s.n+=1
  if s.n==0
    s.m = x
    s.s = 0.
    mean = x
  else
    mean = s.m + (x - s.m) / s.n
    sd = s.s + (x - s.m)*(x - mean)
    s.m = mean
    s.s = sd
  end
end

function var(s::GaussianStream)
  s.n > 1 ? s.s / (s.n-1) : 0.0
end

function sd(s::GaussianStream)
  sqrt(var(s))
end

function Normal(s::GaussianStream)
  stdev = sd(s)
  Normal(s.m, stdev==0 ? 1 : stdev)
end

function kl(p::GaussianStream, q::GaussianStream)
  σ1 = sd(p)
  σ2 = sd(q)
  log(σ2/σ1) + ( (σ1^2 + (p.m - q.m)^2) / (2*(σ2^2)) ) - 0.5
end

# Estimate the number of examples we would need
# to achieve full density
function estExtra( model::DiscCount )
  if model.n > 0
    return log(2, model.n) * (length(model.conditional) - nnz(model.conditional))
  else
    return 0.0
  end
end

function estPMin( model::DiscCount, extra::Float64 )
  1.0 / (model.n + extra)
end

function update(attrs::IntSet, probs::Probability, counts::Counting )
  shape = counts.shape
  probs.label = ./(counts.label, counts.n)
  probs.disc.feature = ./(counts.disc.feature, counts.n)

  extraN = estExtra( counts.disc )

  for a in attrs
    attrID = shape.index[a]
    if shape.isReal[a]

      probs.normal.feature[attrID] = Normal(counts.normal.feature[attrID])
      for c=1:shape.labels
        probs.normal.conditional[c,attrID] = Normal(counts.normal.conditional[c,attrID])
      end

    else

      for c=1:shape.labels
        ca = counts.disc.conditional[c, attrID] / (counts.label[c] + extraN*probs.label[c])
        probs.disc.conditional[ c, attrID ] = isfinite(ca) ? ca : 0
      end

    end
  end
end

function push!(row::Data.Row, model::Counting)
  model.n += 1
  for c in row.labels
    model.label[c] += 1
    for v in row.values
      push!(model, c, v)
    end
  end
end

# Count to fit a Normal Distribution

function merge!(dest::DiscCount, src::DiscCount)
  for i=1:length(src.label)
    dest.label[i] += src.label[i]
  end

  broadcast!(+, dest.feature, dest.feature, src.feature)

  I,J,V = findnz(src.conditional)
  for (i,j,v) in zip(I,J,V)
    dest.conditional[i,j] += v
  end

  return dest
end

function merge!(dest::NormalCount, src::NormalCount)
  broadcast!(+,dest.features, dest.features, src.features)
  broadcast!(+,dest.conditional, dest.conditional, src.conditional)
end

function merge!(dest::Counting, src::Counting)
  merge!(dest.disc, src.disc)
  merge!(dest.normal, src.normal)

  broadcast!(+, dest.label, dest.label, src.label)
  dest.n += src.n
end

export get, merge!, push!, update, var, sd, estExtra, estPMin, Normal
end # module
