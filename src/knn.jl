module KNN

using ..Common
using ..Model
using ..Data
using ..Measures

InitStage = 1
IteratingStage = 1

# Implementation of:
#    Angiulli, Fabrizio. "Fast nearest neighbor condensation for large data sets classification."
#    Knowledge and Data Engineering, IEEE Transactions on 19.11 (2007): 1450-1464.

type Params
    k::Int64
    kMetric::Float64
    destPath::String
    
    Params(params::Dict) = new( get(params,"k",1),
                                get(params,"kMetric",1),
                                get(params, "dest", "") )
end

type Subset
    count::Model.Counting
    subsetcount::Model.Counting
    subset::Vector{Data.Row}
    stage::Int64
    params::Params
    Subset(shape::Data.Shape, params::Dict) = 
        new(Model.Counting(shape),
              Model.Counting(shape),
              [],
              InitStage,
              Params(params) )
end

function eachTask(S::Vector{Data.Row})
  for r in S
    produce(r)
  end
end

function each(S::Vector{Data.Row})
  @task eachTask(S)
end

function getCentroid(model::Model.Counting, class::Int64)
  r = Data.Row(IntSet(class), 0, model.n + class, [])
  cn = 1/model.classcount[class]

  for i=1:size(model.count,2)
    mi = model.count[class,i]
    if mi != 0
      push!(r.values, (i,mi*cn))
    end
  end

  return r
end

function getNN(point::Data.Row,
                  cloud::Vector{Data.Row},
                  queue::Common.TopKQueue,
                  cache::SparseMatrixCSC{Float64,Int64},
                  k::Float64)
  for p in cloud
    push(queue, p, getDist(p,point,cache,k))
  end
  return queue
end


function getNN(point::Data.Row,
                  cloud::Vector{Data.Row},
                  queue::Common.TopKQueue,
                  k::Float64)
  for p in cloud
    push(queue, p, dist(p,point,k))
  end
  return queue
end

function getNN(point::Data.Row,
                  cloud::Vector{Data.Row},
                  start::Int64,
                  finish::Int64,
                  queue::Common.TopKQueue,
                  k::Float64)
  for i=start:finish
    p = cloud[i]
    push(queue, p, dist(p, point, k))
  end
  return queue
end

function labelNN(point::Data.Row,
                   queue::Common.TopKQueue,
                   cache::SparseMatrixCSC{Float64,Int64})

  labelDist = Dict{Int64,Float64}()
  labelCnt = Dict{Int64,Int64}()

  for (row,d) in each(queue)
    # Get the total distance to the label
    for cls in row.labels
      labelDist[cls] = get(labelDist,cls,0) + d
      labelCnt[cls] = get(labelCnt,cls,0) + 1
    end
  end

  nearest = 0
  nearestDist = Inf

  for (k,v) in labelDist
    v /= labelCnt[k]
    if v < nearestDist
      nearestDist=v
      nearest = k
    end
  end

  return nearest
end

function getDist(a::Data.Row,
                  b::Data.Row,
                  cache::SparseMatrixCSC{Float64,Int64},
                  k::Float64)

    if a.num < b.num
        c = a; a = b; b = c
    end

    d = cache[ a.num, b.num ]
    if d == 0
        d = dist(a, b, k)
        cache[ a.num, b.num ] = d
    end
    return d

end

#
# Recursively finds the boundry between items 
# which are > v and those which are not...
#
function find(values::Vector{Float64}, v::Float64, first::Int64, last::Int64)

  if (first + 1) >= last
    return first
  else

    mid = convert(Int64,floor((first + last)/2))

    if values[mid] > v
      return find(values, v, first, mid)
    else
      return find(values, v, mid, last)
    end
    
  end

end

function find(values::Vector{Float64}, v::Float64)
  find(values, v, 0, length(values))
end

function train(model::Subset, data::Data.Dataset)

  if model.stage == InitStage
    Model.count(data, model.count, true)
    model.stage=IteratingStage
  end

  center = Data.vecAsRow( prob.pfeature )
  const kMetric = model.params.kMetric

  nearest = [ Common.TopKQueue(>, model.params.k) for i=1:model.n ]
  nclasses = size(model.classcount, 1)
  S = model.subset
  centroids = [ getCentroid(model, i) for i=1:nclasses ]
  SDelta = centroids

  haveRows = IntSet()
  distCache = spzeros(model.n, model.n)

  while length(SDelta) > 0

    for i=1:length(SDelta)
      countRow(SDelta[i], model::subsetcount)
      push!(S, SDelta[i])
    end
    
    sort!(model.subset, by=(x)-> dist(center, x, model.params.k))

    prevSDelta = Data.Row[]
    SDelta=Data.Row[]
    rep = Dict{Int64,Data.Row}

    # for each (q in (T-S))
    for query in Data.eachrow(data)
      if !in(query.num, haveRows)

          for p in SDelta
              if p.num != query.num
                  push(nearest[query.num], p, getDist(query, p, distCache, model.params.kMetric))
              end
          end

          # Is this misclassified?
          if !in(labelNN(query, nearest[query.num], distCache), query.labels)

            for (neigh,nDist) in each(nearest[query.num])
                if intersection(neigh.labels, query.labels) == 0
                  nrep = get(rep, neigh.num, nothing)
                  if nrep == nothing || nDist < getDist(neigh, nrep, distCache, model.params.kMetric)
                    rep[neigh.num] = query
                  end
                end
            end

          end

      end
    end

    for p in S
      if haskey(rep, p.num)
          push!(SDelta, rep[p.num])
      end
    end

  end

  if length(model.params.destPath) > 0
    Data.write(data, each(S), model.params.destPath)
  end

  return model
end

function label(model::Subset, params::Dict, stream::Task)

  prob = Model.Probability(model.subsetcount, 0.0)
  center = Data.vecAsRow(prob.pfeature)
  sort!(model.subset, by=(x)-> Measures.dist(center, x, model.params.k))
  subdist = [ Measures.dist(center, x, model.params.k) for k in model.subset ]

  for r in stream
    neigh = Common.TopKQueue(>, model.params.k)
    distCenter = Measures.dist(center, r, model.params.k)
    produce( 
        labelNN(r, 
            getNN(r, 
                model.subset,
                find(subdist, distCenter/2), # low end
                find(subdist, distCenter*2), # high end
                neigh,
                model.params.kMetric)))
  end

    return model
end

export train, label, Subset
end # module
