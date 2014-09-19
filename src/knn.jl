module KNN

using ..Common
using ..Model
using ..Data
using ..Measures


InitStage = 1
IteratingStage = 1

typealias RowQ Collections.PriorityQueue{Data.Row, Float64}

# Implementation of:
#    Angiulli, Fabrizio. "Fast nearest neighbor condensation for large data sets classification."
#    Knowledge and Data Engineering, IEEE Transactions on 19.11 (2007): 1450-1464.

type Params
  k::Int64
  kMetric::Float64
  maxSize::Int64

  function Params(params::Common.Params)
    new( get(params,"k",10),
        get(params,"kMetric",2),
        get(params, "max", 10000), )
  end
end

type Subset
  shape::Data.Shape
  count::Model.Counting
  subsetcount::Model.Counting
  subset::Vector{Data.Row}
  stage::Int64
  params::Params
  function Subset(shape::Data.Shape, params::Common.Params)
    new(shape,
        Model.Counting(shape),
        Model.Counting(shape),
        [],
        InitStage,
        Params(params) )
  end
end

function eachTask(S::Vector{Data.Row})
  for r in S
    produce(r)
  end
end

function each(S::Vector{Data.Row})
  @task eachTask(S)
end

function getCentroid(count::Model.Counting, class::Int64)
  r = Data.Row(IntSet(class), 0, count.n + class, Data.Value[])
  cn = 1/count.classcount[class]

  for i=1:size(count.overlap,2)
    mi = count.overlap[class,i]
    if mi != 0
      push!(r.values, (i,mi*cn))
    end
  end

  return r
end

function getNN(point::Data.Row,
               cloud::Vector{Data.Row},
               queue::RowQ,
               cache::SparseMatrixCSC{Float64,Int64},
               kMetric::Float64,
               k::Int64)
  for p in cloud
    update!(queue, p, dist(p,point,kMetric), k)
  end
  return queue
end

function getNN(point::Data.Row,
               cloud::Vector{Data.Row},
               queue::RowQ,
               cache::SparseMatrixCSC{Float64,Int64},
               params::Params)
  getNN(point, cloud, queue, cache, params.kMetric, params.k)
end


function getNN(point::Data.Row,
               cloud::Vector{Data.Row},
               queue::RowQ,
               kMetric::Float64,
               k::Int64)
  for p in cloud
    update!(queue, p, dist(p,point,kMetric), k)
  end
  return queue
end

function getNN(point::Data.Row,
               cloud::Vector{Data.Row},
               start::Int64,
               finish::Int64,
               queue::RowQ,
               kMetric::Float64,
               k::Int64)
  for i=start:finish
    p = cloud[i]
    update!(queue, p, dist(p, point, kMetric), k)
  end
  return queue
end

function labelNN(point::Data.Row, queue::RowQ)

  labelDist = Dict{Int64,Float64}()
  labelCnt = Dict{Int64,Int64}()

  for (row,d) in queue
    # Get the total distance to the label
    for cls in row.labels
      labelDist[cls] = get(labelDist,cls,0) + abs(d)
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

function update!(q::RowQ, row::Data.Row, dist::Float64, k::Int64)
  q[row] = -dist
  while length(q) > k
    Collections.dequeue!(q)
  end
end

function dist(a::Data.Row, b::Data.Row, k::Float64)

  d = 0.0
  const imax = length(a.values)
  const jmax = length(b.values)
  i=1
  j=1

  while i <= imax && j <= jmax
    ai,av = a.values[i]
    bi,bv = b.values[j]

    if ai < bi
      i += 1
      d += abs(av)^k
    elseif ai > bi
      j += 1
      d += abs(bv)^k
    else
      i += 1
      j += 1
      try
        d += abs(av - bv)^k
      catch
        println("(",av," - ", bv, ") ^ ", k)
        exit()
      end
    end
  end

  while i <= imax
    ai,av = a.values[i]
    i += 1
    d += abs(av)^k
  end

  while j <= jmax
    bi,bv = b.values[j]
    j += 1
    d += abs(bv)^k
  end

  return d^(1/k)
end

function dist(a::Data.Row, b::Data.Row, k::Float64, cache)
  cdist = cache[a.num,b.num]
  if cdist > 0
    return cdist
  else
    return cache[a.num,b.num] = dist(a,b,k)
  end
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
  find(values, v, 1, length(values))
end

function train(model::Subset, data::Data.Dataset)

  if model.stage == InitStage
    Model.count(Data.eachrow(data), model.count, false)
    model.stage=IteratingStage
  end

  prob = Model.Probability(model.count)

  center = Data.vecAsRow( prob.pfeature )
  const kMetric = model.params.kMetric

  nearest = [ RowQ() for i=1:model.count.n ]
  q_center = [ (1, 0.0) for i=1:model.count.n ]

  nclasses = size(model.count.classcount, 1)
  S = model.subset
  centroids = [ getCentroid(model.count, i) for i=1:nclasses ]
  nearest_centers = [ RowQ() for i=1:nclasses ]
  distCache = spzeros(model.count.n + nclasses, model.count.n)

  SDelta = SDelta=Data.Row[]

  println(STDERR,"Finding $(model.params.k) central examples for each class")
  for query in Data.eachrow(data)
    qkMin = Inf
    for k=1:nclasses

      qkDist = dist(query, centroids[k], model.params.kMetric)
      if qkDist < qkMin
        qkMin = qkDist
        q_center[query.num] = (k, qkDist)
      end

      update!(nearest_centers[k], query, qkDist, model.params.k)
    end
  end

  for i=1:nclasses
    for (n,d) in nearest_centers[i]
      push!(SDelta, n)
    end
  end

  haveRows = IntSet()

  while length(SDelta) > 0 && length(S) < model.params.maxSize

    println(STDERR, "Next run, SDelta length: ",length(SDelta), )

    for i=1:length(SDelta)
      push!(S, SDelta[i])
      push!(haveRows, SDelta[i].num)
    end

    refDelta = [ SDelta[1:end] for i=1:nclasses ]

    for i=1:nclasses
      sort!(refDelta[i], by=(x)-> dist(centroids[i], x, model.params.kMetric, distCache))
    end

    rep = Dict{Int64,(Data.Row,Float64)}()

    comps = 0

    # for each (q in (T-S))
    for query in Data.eachrow(data)
      if !in(query.num, haveRows)
        qNeighbors = nearest[query.num]
        qGroup, qDist = q_center[query.num]
        dMax = qDist * 2

        for p in refDelta[qGroup]
          if p.num != query.num
            comps += 1
            update!(qNeighbors, p,
                    dist(query, p, model.params.kMetric),
                    model.params.k)
          end
        end

        # Is this misclassified?
        if !in(labelNN(query, qNeighbors), query.labels)

          for (neigh,nDist) in qNeighbors
            if length(intersect(neigh.labels, query.labels)) == 0

              nDist=abs(nDist)
              if !haskey(rep, neigh.num) || nDist < rep[neigh.num][2]
                rep[neigh.num] = (query, qDist)
              end

            end
          end

        end
      end
    end

    SDelta=Data.Row[]

    for p in S
      if haskey(rep, p.num)
        push!(SDelta, rep[p.num][1])
      end
    end

  end

  println(STDERR,"Subset of size ", length(S), " found")

  return model
end

function label(model::Subset, params::Common.Params, stream::Task)

  prob = Model.Probability(model.subsetcount)
  center = Data.vecAsRow(prob.pfeature)

  sort!(model.subset, by=(x)-> dist(center, x, model.params.kMetric))
  subdist = [ dist(center, x, model.params.kMetric) for x in model.subset ]
  sublen = length(subdist)

  for r in stream
    neigh = RowQ()
    distCenter = dist(center, r, model.params.kMetric)

    produce(
      (labelNN(r,
               getNN(r,
                     model.subset,
                     find(subdist, clamp(distCenter/2, 1, sublen)), # low end
                     find(subdist, clamp(distCenter*2, 1, sublen)), # high end
                     neigh,
                     model.params.kMetric,
                     model.params.k)), r.labels))
  end

  return model
end

export train, label, Subset
end # module
