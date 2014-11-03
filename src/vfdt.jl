module VFDT

import ..Common
import ..Data
import ..Model
import ..NaiveBayes
import DataStructures: Deque
import ..Measures: ConfusionMatrix, gain, gini,
precision,
recall,
sensitivity,
specificity,
npv,
fallout,
falsediscovery,
missrate,
accuracy,
f1,
mcc,
kl,
informedness,
markedness,
dist,
bhattacharyya

using Distributions

typealias SparseMat SparseMatrixCSC{Float64,Int64}

type TreeNode
  id::Int64
  isNull::Bool
  attr::Int64
  bestG::Float64
  epsilon::Float64
  children::Vector{TreeNode}
  parent::TreeNode
  shape::Data.Shape
  m::Model.Counting
  p::Model.Probability
  nb::NaiveBayes.NBPredModel
  queue::Deque{Data.Row}

  ignorePriors::Bool

  depth::Int64
  height::Int64
  label::Int64
  ttl::Int64

  traffic::Int64

  updatedAttrs::IntSet
  ranks::Vector{Common.Ranking}

  function TreeNode()
    self = new()
    self.id=0
    self.isNull = true
    self.depth=0
    self.height=0
    self.traffic=0
    self.children = TreeNode[]
    self.epsilon=0
    self.bestG=0
    return self
  end

  TreeNode(shape::Data.Shape) = TreeNode(TreeNode(), 0, shape, false)
  TreeNode(shape::Data.Shape, id::Int64) = TreeNode(TreeNode(), id, shape, false)
  TreeNode(shape::Data.Shape, ignorePriors::Bool) = TreeNode(TreeNode(), 0, shape, ignorePriors)
  TreeNode(shape::Data.Shape, id::Int64, ignorePriors::Bool) = TreeNode(TreeNode(), id, shape, ignorePriors)

  function TreeNode(parent::TreeNode, id::Int64, shape::Data.Shape, ignorePriors::Bool)
    self = TreeNode()
    self.id = id
    self.parent = parent
    self.depth = parent.depth + 1
    self.shape = shape
    self.ignorePriors = ignorePriors
    self.m = Model.Counting(shape)
    reset(self)
    self.isNull = false
    return self
  end
end

typealias NodeQueue Collections.PriorityQueue{TreeNode, Float64}

type Params
  ttl::Int64
  pruneEvery::Int64
  delta::Float64
  tieLimit::Float64
  maxNodes::Int64
  quantiles::Vector{Float64}
  naivebayes::Bool
  ignorePriors::Bool
  iters::Int64
  churn::Int64
  graphFile::String
  G::Function

  function Params(params::Common.Params)
    G = eval( symbol( get(params,"measure","gain") ) ) # NOT USED
    new(
      get(params, "ttl", 500),
      get(params, "pruneEvery", 50),
      get(params, "delta", 0.1),
      get(params, "tieLimit", 0.05,),
      clamp(get(params, "maxNodes", 4096), 8, 65536), # Change this if you need...
      get(params, "quantiles", [0.25,0.75]),
      get(params, "naivebayes", true,),
      get(params, "ignorePriors", true,),
      get(params, "iters", 3,),
      get(params, "churn", 1000,),
      get(params, "graphFile", "",),
      G )
  end
end

type HoeffdingTree
  shape::Data.Shape
  cost::NodeQueue
  size::Int64
  root::TreeNode
  labelDist::Vector{Float64}
  params::Params

  function HoeffdingTree(shape::Data.Shape, params::Common.Params)
    tree = new()
    tree.params = Params(params)
    tree.shape = shape
    tree.cost = NodeQueue()
    tree.size = 0
    tree.root = TreeNode(shape, tree.params.ignorePriors)
    tree.labelDist = zeros( shape.labels )
    return tree
  end

end

function inc(tree::HoeffdingTree)
  tree.size += 1
end

function isLeaf(n::TreeNode)
  length(n.children)==0
end

function reset(self::TreeNode)
  reset(self, self.shape)
end

function reset(self::TreeNode, shape::Data.Shape)

  self.epsilon=0
  self.isNull = true
  self.attr = -1
  self.children = TreeNode[]
  self.m = Model.Counting(shape)
  self.p = Model.Probability(self.m)

  self.height = 0
  self.label = 0
  self.ttl = 0

  self.updatedAttrs = IntSet()
  self.ranks = Vector{Common.Ranking}[]
  return self

end

function print(tree::HoeffdingTree)
  print(STDOUT, tree)
end

function print(S::IO, tree::HoeffdingTree)
  println(S,"digraph G{")
  print(S, tree.root,"T")
  println(S,"}")
end

function print(S::IO, node::TreeNode, path::String)
  if !node.isNull

    if isLeaf(node)
      node.label = indmax( node.m.label )

      println(S, path, " [label=\"", node.shape.labelStrings[ node.label ],"\" weight=\"", node.traffic,"\"];")
    else
      println(S, path, " [label=\"", node.shape.names[ node.attr ],":", node.traffic,"\" weight=\"", node.traffic,"\"];")
    end

    i=0
    for c in node.children
      i += 1
      cn = "$(path)_$(i)"
      print(S, c, cn)
      println(S, path, " -> ", cn, " [ label = $i];")
    end
  end
end

function visitParents(node::TreeNode)
  if !node.parent.isNull
    produce(node.parent)
    visitParents(node.parent)
  end
end

function visitChildren(node::TreeNode)
  @task visitChildrenTask(node)
end

function visitChildrenTask(node::TreeNode)
  for c in node.children
    produce(c)
  end

  for c in node.children
    visitChildrenTask(c)
  end
end

function visitLeaves(node::TreeNode)
  @task visitLeavesTask(node)
end

function visitLeavesTask(node::TreeNode)
  if isLeaf(node)
    produce(node)
  else
    for c in node.children
      visitLeavesTask(c)
    end
  end
end

function getVisitor(node::TreeNode)
  return @task visitChildren(node)
end

function getLeafVisitor(node::TreeNode)
  return @task visitLeaves(node)
end

function getParentVisitor(node::TreeNode)
  return @task visitParents(node)
end

function getChildIndex(shape::Data.Shape, v::Data.DiscValue)
  v.value - shape.offsets[v.index]
end

function getChildIndex(dist::Model.Normal, q::Vector{Float64}, v::Data.RealValue)
  qval = cdf( dist, v.value )

  if qval > 0.5

    i = length(q) + 1
    while qval < q[i-1]
      i-=1
    end

  else

    i=1
    while qval > q[i]
      i+=1
    end

  end

  return i
end

function isLeaf(node::TreeNode)
  length(node.children) == 0
end

function rowIndex(row::Data.Row)
  vec = Dict{Int64, Data.Value}()

  for p in row.values
    vec[ p.feature ] = p
  end

  return vec
end

function sortToLeaf(node::TreeNode, q::Vector{Float64}, rIndex::Dict{Int64,Data.Value})

  node.traffic += 1

  if length(node.children)==0
    return node
  else

    if haskey( rIndex, node.attr )

      v = rIndex[node.attr]
      if isa(v, Data.RealValue)
        idx = getChildIndex( node.p.normal.feature[ v.index ], q, v )
      else
        idx = getChildIndex( node.p.shape, v )
      end

      return sortToLeaf(node.children[idx], q, rIndex)
    else
      return sortToLeaf(node.children[1], q, rIndex)
    end

  end
end

function countToLeaf!(tree::HoeffdingTree, row::Data.Row)
  countToLeaf!(tree.root, tree.params.quantiles, row, rowIndex(row))
end

function countToLeaf!(node::TreeNode, q::Vector{Float64}, row::Data.Row, rIndex::Dict{Int64,Data.Value})

  node.traffic += 1

  Model.push!(row, node.m)

  if length(node.children)==0
    return node
  else

    if haskey( rIndex, node.attr )

      v = rIndex[node.attr]
      if isa(v, Data.RealValue)
        idx = getChildIndex( node.p.normal.feature[ v.index ], q, v )
        assert(idx != 0)
      else
        idx = getChildIndex( node.p.shape, v )
        assert(idx != 0)
      end

      return countToLeaf!(node.children[idx], q, row, rIndex)
    else
      return countToLeaf!(node.children[1], q, row, rIndex)
    end
  end
end


function setMeasures(tree::HoeffdingTree, node::TreeNode)
  tree.cost[node] = node.bestG * node.traffic
end

function clearMeasures(tree::HoeffdingTree, node::TreeNode)
  tree.goodness[node] = -1
  tree.cost[node] = -1
end

function splitLeaf(tree::HoeffdingTree, node::TreeNode, onAttr::Int64, epsilon::Float64)

  if !node.parent.isNull
    clearMeasures(tree, node.parent)
  end
  setMeasures(tree, node)

  if tree.shape.isReal[ onAttr ]
    nChildren = length(tree.params.quantiles) + 1
  else
    nChildren = tree.shape.widths[ tree.shape.index[onAttr] ]
  end

  node.children = [ TreeNode(tree.shape, inc(tree), tree.params.ignorePriors) for i=1:nChildren ]

  # Clean out old objects
  node.p = Model.Probability( Model.Counting( node.shape ) )
  node.updatedAttrs = IntSet()
  node.height = 1

  for n in getParentVisitor(node)
    n.height = maximum([ c.height for c in c.children ]) + 1
  end

  node.attr = onAttr
  node.epsilon = epsilon

end

function prune!(tree::HoeffdingTree)
  cnt = 0
  prune!(tree.root)
  for n in visitChildren(tree.root)
    cnt += 1
    n.id = cnt
  end
  tree.size = cnt
end

function prune!(node::TreeNode)

  if !isLeaf(node)

    trafficBalance = [ c.traffic for c in node.children ]

    if maximum(trafficBalance) == sum(trafficBalance)
      # No real decision is being made here
      for c in node.children
        reset(c)
        c.parent = TreeNode()
      end

      node.children = TreeNode[]
      node.attr = 0
      node.label = indmax(node.m.label)
    else
      for c in node.children
        prune!(c)
      end
    end

  end

end


#
#   This is the inner loop of the VFDT where we decide to split the leaf
#       if needed.
#
function updateLeaf(tree::HoeffdingTree,
                    leaf::TreeNode,
                    ttl::Float64,
                    params::Params,
                    force::Bool )

  shape = tree.shape
  leaf.ttl += 1

  if leaf.ttl >= ttl && length(leaf.updatedAttrs) > 1

    treeDist = ./(tree.labelDist, sum(tree.labelDist))

    leaf.label = indmax(leaf.p.label)
    leaf.ttl = 0
    Model.update(leaf.updatedAttrs, leaf.p, leaf.m)

    # Now loop over attributes and find the best 2 attributes
    bestAttr = 0
    nextBestAttr = 0
    bestG = 0
    nextG = 0
    nQ = length( tree.params.quantiles )
    labelRng = 1:tree.shape.labels

    pLabel = ./(leaf.m.label, sum(leaf.m.label))
    labelWeight = .-(1, pLabel)
    attrWeights = (Float64,Int64,)[]

    for n in leaf.updatedAttrs
      nID = shape.index[n]

      if shape.isReal[n]
        attrDist = [ Model.kl(leaf.m.normal.conditional[c,nID],
                              leaf.m.normal.feature[nID])
                    for c=labelRng ]
      else

        iRng = shape.ranges[nID]
        attrDist = [ kl(leaf.m.disc.conditional[c,iRng],
                        leaf.m.disc.feature[iRng], true)
                    for c=labelRng ]

      end

      broadcast!((x)-> isfinite(x) ? x : 0.0, attrDist, attrDist)
      push!( attrWeights, (sum( .*(labelWeight, attrDist) ), n,) )
    end

    sort!(attrWeights, rev=true)

    bestG = attrWeights[1][1]
    bestAttr = attrWeights[1][2]

    epsilon = sqrt( log(1/params.delta) / (2 * leaf.m.n)  )
    gradient = bestG - attrWeights[2][1]

    leaf.epsilon = epsilon
    leaf.bestG = bestG

    if gradient > params.delta
      splitLeaf(tree, leaf, bestAttr, epsilon)
      return true
    end

  end

  return false
end

function updateLeaf(tree::HoeffdingTree, node::TreeNode, ttl::Float64, params::Params)
  updateLeaf(tree, node, ttl, params, false)
end

function updateLeaf(tree::HoeffdingTree, node::TreeNode, params::Params)
  updateLeaf(tree, node, params.ttl, params, false)
end

#
#   Allows us to simply use the tree for clustering...
#
function routeAndCount(stream::Task, tree::HoeffdingTree)

  for r in stream
    l = sortToLeaf(tree.root, tree.params.quantiles, rowIndex(r))
    Model.push!(r, l.m)
  end

end

function ttl_decay(t)
  1000 * (e ^ -(t/(e^6)))
end

# Training is simple
function train(tree::HoeffdingTree, data::Data.Dataset)

  ttl = float(tree.params.ttl)
  treeTooBig = false
  updateCounter = 0

  for iterNum=1:tree.params.iters

    rowProvider = Data.eachrow(data)
    if tree.params.churn > 0
      rowProvider = Data.churn(rowProvider, tree.params.churn)
    end

    for row in rowProvider

      for n in row.labels
        tree.labelDist[n] += 1
      end

      leaf = countToLeaf!(tree, row)
      for v in row.values
        push!(leaf.updatedAttrs, v.feature)
      end

      updateCounter += int(updateLeaf(tree, leaf, ttl, tree.params))

      if updateCounter > tree.params.pruneEvery
        prune!(tree)
        updateCounter=0
      end

    end

    println(STDERR, "Done with iter $(iterNum)")
  end

  println(STDERR, "Building stats, tree size ", tree.size)
  routeAndCount(Data.eachrow(data), tree)

  if length(tree.params.graphFile) > 0
    open(tree.params.graphFile, "w") do f
      print(f, tree)
      println(STDERR, "Graphviz file written")
    end
  end

  return tree
end

function label(tree::HoeffdingTree, params::Common.Params, stream::Task)

  const naiveBayesLeaves = tree.params.naivebayes
  const produceRanks = get(params, "ranks", false)

  #   Setup the leaves for prediction
  for leaf in visitLeaves(tree.root)
    leaf.p = Model.Probability(leaf.m)
    if naiveBayesLeaves
      leaf.nb = NaiveBayes.NBPredModel(leaf.m, tree.params.ignorePriors)
    end
    leaf.label = indmax(leaf.p.label)

    if produceRanks
      rnks = map((i) -> Common.Ranking(i, leaf.p.label), 1:size(leaf.shape.labels))
      sort!(rnks, by=(x) -> -x.value)
      leaf.ranks = rnks
    end
  end

  for row in stream
    row_leaf = sortToLeaf( tree.root, tree.params.quantiles, rowIndex(row) )

    if naiveBayesLeaves
      produce( (NaiveBayes.label(row_leaf.nb, row, produceRanks), row.labels) )
    elseif produceRanks
      produce( (row_leaf.ranks, row.labels)  )
    else
      produce( (row_leaf.label, row.labels)  )
    end

  end

end

export train, label, HoeffdingTree
end # module
