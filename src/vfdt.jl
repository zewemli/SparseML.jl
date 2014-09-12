module VFDT

import ..Common
import ..Data
import ..Model
import ..NaiveBayes
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
                    informedness,
                    markedness


type TreeNode
    id::Int64
    isNull::Bool
    isLeaf::Bool
    attr::Int64
    bestG::Float64
    epsilon::Float64
    objective::Float64
    low::TreeNode
    high::TreeNode
    parent::TreeNode
    datashape::Data.Shape
    m::Model.Counting
    p::Model.Probability
    nb::NaiveBayes.NBPredModel
    confusion::Matrix{Float64}

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

      self.epsilon=0
      self.bestG=0
      self.objective=0
      return self
    end

    TreeNode(shape::Data.Shape) = TreeNode(TreeNode(), 0, shape, false)
    TreeNode(shape::Data.Shape, ignorePriors::Bool) = TreeNode(TreeNode(), 0, shape, ignorePriors)
    function TreeNode(parent::TreeNode, id::Int64, shape::Data.Shape, ignorePriors::Bool)
        self = TreeNode()
        self.id = id
        self.parent = parent
        self.depth = parent.depth + 1
        self.datashape = shape
        self.ignorePriors = ignorePriors
        reset(self)
        self.isNull = false
        return self
      end
end

function reset(self::TreeNode)
  reset(self, self.datashape)
end

function reset(self::TreeNode, shape::Data.Shape)

  self.epsilon=0
  self.objective=0
  self.isNull = true
  self.isLeaf = true
  self.attr = -1
  self.low = TreeNode()
  self.high = TreeNode()
  self.m = Model.Counting(shape)
  self.p = Model.Probability(self.m)
  self.nb = NaiveBayes.NBPredModel( self.p, self.ignorePriors )
  self.confusion = zeros(shape.classes, shape.classes)

  self.height = 0
  self.label = 0
  self.ttl = 0

  self.updatedAttrs = IntSet()
  self.ranks = Vector{Common.Ranking}[]
  return self

end

function imbalance(m::Model.Probability, attr::Int64)

  majority = indmax(m.pclass)
  return m.pclass[majority] - m.prob[majority,attr]

end

typealias NodeQueue Collections.PriorityQueue{TreeNode, Float64}

type Params
    ttl::Int64
    delta::Float64
    tieLimit::Float64
    maxNodes::Int64
    contractInterval::Int64
    minTraffic::Float64
    naivebayes::Bool
    ignorePriors::Bool
    iters::Int64
    G::Function

    function Params(params::Common.Params)
        G = eval( symbol( get(params,"measure","gini") ) )
        new(
            get(params, "ttl", 10),
            get(params, "delta", 0.005),
            get(params, "tieLimit", 0.005,),
            clamp(get(params, "maxNodes", 512), 8, 65536), # Change this if you need...
            get(params, "contractInterval", 1000),
            get(params, "minTraffic", 60),
            get(params, "naivebayes", true,),
            get(params, "ignorePriors", true,),
            get(params, "iters", 3,),
            G )
    end
end

type HoeffdingTree
    shape::Data.Shape
    goodness::NodeQueue
    cost::NodeQueue
    counter::Int64
    root::TreeNode
    params::Params
    function HoeffdingTree(shape::Data.Shape, params::Common.Params)
      tree_params = Params(params)
      root = TreeNode(shape, tree_params.ignorePriors)
      root.isNull = false
      new(shape, NodeQueue(), NodeQueue(), 0, root, tree_params)
    end
end
function nodeLabel(node)
    top = 1
    topCount=0
    for i=1:length(m.classcount)
        if m.classcount[i] > topCount
            topCount = m.classcount[i]
            top = i
        end
    end
end

function print(tree::HoeffdingTree)
  println("digraph G{")
  print(tree.root,"")
  println("}")
end

function print(node::TreeNode, prefix::String)
  if !node.isNull
    println(node.id, " [label=\"", node.id,":", node.traffic,"\" weight=\"", node.traffic,"\"];")
    if !node.isLeaf
      print(node.low,"0:")
      print(node.high,"1:")
      println(node.id, " -> ", node.high.id, ";")
      println(node.id, " -> ", node.low.id, ";")
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
    if !node.isLeaf
        produce(node.low)
        produce(node.high)
        if !node.low.isNull
            visitChildren(node.low)
        end
        if !node.high.isNull
            visitChildren(node.high)
        end
    end
end

function visitLeaves(node::TreeNode)
    if node.isLeaf
        produce(node)
    else
        visitLeaves(node.low)
        visitLeaves(node.high)
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

function sortToLeaf(node, vec)

    node.traffic += 1

    if node.isLeaf
        return node
    else
        assert(!node.low.isNull)
        assert(!node.high.isNull)

        if vec[node.attr] == 0.0
            return sortToLeaf(node.low,  vec)
        else
            return sortToLeaf(node.high, vec)
        end
    end
end

function prevAttrs(node::TreeNode)
    r = IntSet()
    p = node.parent
    while !p.isNull
        push!(r, p.attr)
        p = p.parent
    end
    return r
end

function objective(tree::HoeffdingTree, node::TreeNode)

  const ignorePriors = tree.params.ignorePriors

  g = 0.0

  real_col = sum(node.confusion, 1)
  pred_row = sum(node.confusion, 2)

  for i=1:tree.shape.classes
    M = ConfusionMatrix()
    M.tp = node.confusion[i,i]
    M.tn = sum(node.confusion) - M.tp
    M.fp = pred_row[i] - M.tp
    M.fn = real_col[i] - M.tp
    fi = tree.params.F(M)
    g += ignorePriors ? fi : fi *  node.p.pclass[i]
  end
  return g/tree.shape.classes * log(node.traffic)
end

function setMeasures(tree::HoeffdingTree, node::TreeNode)
  tree.goodness[node] = node.objective * node.traffic
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

    # Clean out old objects
    node.p = Model.Probability( Model.Counting( node.datashape ) )
    node.updatedAttrs = IntSet()
    node.height = 1

    for n in getParentVisitor(node)
        n.height = max(n.low.height, n.high.height) + 1
    end

    node.isLeaf = false
    node.attr = onAttr
    node.epsilon = epsilon

    tree.counter += 1
    node.low  = TreeNode(node, tree.counter, node.datashape, tree.params.ignorePriors)
    tree.counter += 1
    node.high = TreeNode(node, tree.counter, node.datashape, tree.params.ignorePriors)

    node.low.depth = node.depth + 1
    node.high.depth = node.depth + 1

end

function trafficBalance(node::TreeNode)
  n = node.low.traffic + node.high.traffic
  2*min(node.low.traffic + node.high.traffic) / n
end

function contract!(tree::HoeffdingTree, node::TreeNode, minTraffic::Float64)
  if !node.isLeaf
    if min(node.low.traffic, node.high.traffic) < minTraffic

      if node.low.traffic < node.high.traffic
        collapseNode(tree, node.low)
        rt = node.high
        deadnode = node.low
      else
        collapseNode(tree, node.high)
        rt = node.low
        deadnode = node.high
      end

      contract!(tree, rt, minTraffic)

      if rt.isLeaf
        collapseNode(tree, node)
      else
        Model.merge!(node.m, node.high.m)
        Model.merge!(node.m, node.low.m)
        reset(deadnode)
        node.attr = rt.attr
        node.low = rt.low
        node.low.parent = node
        node.high = rt.high
        node.high.parent = node
        rt.low = TreeNode()
        rt.high = TreeNode()
      end

    else
      contract!(tree, node.low, minTraffic)
      contract!(tree, node.high, minTraffic)
    end
  end
end

function collapseNode(tree::HoeffdingTree, node::TreeNode)

    if !node.isLeaf

        node.updatedAttrs = IntSet()

        if !node.low.isLeaf
          collapseNode(tree, node.low)
          Model.merge!(node.m, node.low.m)
        end

        if !node.high.isLeaf
          collapseNode(tree, node.high)
          Model.merge!(node.m, node.high.m)
        end

        if !node.parent.isNull
          setMeasures(tree, node.parent)
        end
        clearMeasures(tree, node)

        node.attr=0
        node.isLeaf = true

        reset(node.low)
        node.low = TreeNode()
        reset(node.high)
        node.high = TreeNode()

    end
end

function updateLeafPredictions(tree::HoeffdingTree, node::TreeNode, row::Data.Row, params::Params)

  vecLabel = node.label
  if params.naivebayes
    if node.ttl == 0
      # node.p was updated in updateLeaf
      NaiveBayes.update(node.nb, node.p, params.ignorePriors)
    end

    vecLabel = NaiveBayes.label(node.nb, row, false)
  end

  for l in row.labels
    node.confusion[vecLabel, l]
  end

  node.objective = objective(tree,node)
end

#
#   This is the inner loop of the VFDT where we decide to split the leaf
#       if needed.
#
function updateLeaf(tree::HoeffdingTree, leaf::TreeNode, ttl::Float64, params::Params, force::Bool)

    leaf.ttl += 1

    if leaf.ttl >= ttl && leaf.traffic > params.minTraffic

      pc = 0.0
      for i=1:size(leaf.p.pclass,1)
          if leaf.p.pclass[i] > pc
              pc = leaf.p.pclass[i]
              leaf.label = i
          end
      end

      leaf.ttl = 0
      Model.update(leaf.updatedAttrs, leaf.p, leaf.m)
      ignoreAttrs = prevAttrs(leaf)

      # Now loop over attributes and find the best 2 attributes
      bestAttr = 0
      nextBestAttr = 0
      bestG = 0
      nextG = 0

      for n in setdiff(leaf.updatedAttrs, ignoreAttrs)
        nG = params.G(leaf.p, n)

        if nG > bestG
            nextG = bestG
            nextBestAttr = bestAttr
            bestG = nG
            bestAttr = n
        elseif nG > nextG
            nextG = nG
            nextBestAttr = n
        end

      end

      if bestAttr != 0

        epsilon = sqrt( log(1/params.delta) / (2 * leaf.m.n)  )
        gradient = bestG - nextG

        leaf.epsilon = epsilon
        leaf.bestG = bestG

        if gradient > params.delta || gradient < params.tieLimit
            splitLeaf(tree, leaf, bestAttr, epsilon)
        end

      end

    end
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

    for l in @task visitLeaves(tree.root)
        l.m = spzeros(size(l.m,1), size(l.m,2))
    end

    for r in stream
        l = sortToLeaf(tree.root, r)
        Model.countRow(r, l.m)
        Model.countFeatureOverlaps(r, l.m)
    end

end

function ttl_decay(t)
  1000 * (e ^ -(t/(e^6)))
end

# Training is simple
function train(tree::HoeffdingTree, data::Data.Dataset)
    nbins = tree.shape.unique
    tmpVec = zeros( (tree.shape.features+1) * nbins )

    contractionCounter=tree.params.contractInterval

    for iterNum=1:tree.params.iters

      for row in Data.eachrow(data, true)

          fill!(tmpVec,0)
          Model.project!(row, tmpVec, nbins)

          leaf = sortToLeaf(tree.root, tmpVec)
          Model.countRow(row, leaf.m)
          for v in row.values
            push!(leaf.updatedAttrs, Model.project(v, row.nbins))
          end

          updateLeaf(tree, leaf, tree.params.ttl + ttl_decay(row.num), tree.params)
          #updateLeafPredictions(tree, leaf, row, tree.params)

          while length( tree.cost ) > 0 && Collections.peek( tree.cost )[2] < 0
            Collections.dequeue!( tree.cost )
          end

          while length( tree.cost ) > tree.params.maxNodes
            dq = Collections.dequeue!( tree.cost )
            if !dq.isNull && !dq.parent.isNull
              collapseNode(tree, dq)
            end
          end

          contractionCounter -= 1
          if contractionCounter == 0
            contract!(tree, tree.root, tree.params.minTraffic)
            contractionCounter = tree.params.contractInterval
          end
      end
      println(STDERR, "Done with iter $(iterNum)")
    end
    contract!(tree, tree.root, tree.params.minTraffic)

    return tree
end

function label(tree::HoeffdingTree, params::Common.Params, stream::Task)

    nbins = tree.shape.unique
    const naiveBayesLeaves = tree.params.naivebayes
    const produceRanks = get(params, "ranks", false)

    #   Setup the leaves for prediction
    for leaf in getLeafVisitor(tree.root)
        leaf.p = Model.Probability(leaf.m)
        if naiveBayesLeaves
          leaf.nb = NaiveBayes.NBPredModel(leaf.p, tree.params.ignorePriors)
        end
        leaf.label = indmax(leaf.p.pclass)

        if produceRanks
          rnks = map((i) -> Common.Ranking(i, leaf.p.pclass), 1:size(leaf.p.pclass))
          sort!(rnks, by=(x) -> -x.value)
          leaf.ranks = rnks
        end
    end

    rowAsVec = zeros( (tree.shape.features+1) * nbins )
    if naiveBayesLeaves

      for row in stream
        fill!(rowAsVec,0)
        Model.project!(row, rowAsVec, nbins)
        row_leaf = sortToLeaf(tree.root, rowAsVec)
        produce( (NaiveBayes.label(row_leaf.nb, row, produceRanks), row.labels) )
      end

    else

      # Do the actual labelling
      if produceRanks
        for row in stream
          fill!(rowAsVec,0)
          Model.project!(row, rowAsVec, nbins)
          produce( (sortToLeaf(tree.root, rowAsVec).ranks, row.labels)  )
        end
      else
        for row in stream
          fill!(rowAsVec,0)
          Model.project!(row, rowAsVec, nbins)
          produce( (sortToLeaf(tree.root, rowAsVec).label, row.labels)  )
        end
      end
    end
end

export train, label, HoeffdingTree
end # module
