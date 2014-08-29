module VFDT

import ..Common
import ..Data
import ..Model
import ..Measures: gain, gini
import ..NaiveBayes

type TreeNode
    isNull::Bool
    isLeaf::Bool
    attr::Int64
    epsilon::Float64
    low::TreeNode
    high::TreeNode
    parent::TreeNode
    datashape::Data.Shape
    m::Model.Counting
    p::Model.Probability
    nb::NaiveBayes.NBPredModel

    depth::Int64
    height::Int64
    label::Int64
    ttl::Int64

    updatedAttrs::IntSet
    ranks::Vector{Common.Ranking}

    function TreeNode()
      self = new()
      self.isNull = true
      self.depth=0
      self.height=0
      return self
    end

    TreeNode(shape::Data.Shape) = TreeNode(TreeNode(), shape)

    function TreeNode(parent::TreeNode, shape::Data.Shape)

        self = TreeNode()
        self.isNull = false
        self.isLeaf = true
        self.attr = -1
        self.epsilon = 0
        self.low = TreeNode()
        self.high = TreeNode()
        self.parent = parent
        self.datashape = shape
        self.m = Model.Counting(shape)
        self.p = Model.Probability(self.m)
        self.nb = NaiveBayes.NBPredModel( self.p, false )

        self.depth = parent.depth + 1
        self.height = 0
        self.label = 0
        self.ttl = 0

        self.updatedAttrs = IntSet()
        self.ranks = Vector{Common.Ranking}[]

        return self
      end
end

typealias NodeQueue Collections.PriorityQueue{TreeNode, Float64}

type Params
    ttl::Int64
    delta::Float64
    tieLimit::Float64
    maxDepth::Int64
    G::Function

    function Params(params::Dict)
        G = eval( symbol( get(params,"measure","gain") ) )

        new(
            get(params,"ttl",200),
            get(params,"delta",0.005),
            get(params,"tieLimit",0.005,),
            clamp(get(params,"maxDepth",16),2,24), # Change this if you need...
            G )
    end
end

type HoeffdingTree
    shape::Data.Shape
    queue::Collections.PriorityQueue{TreeNode, Float64}
    root::TreeNode
    params::Params
    HoeffdingTree(shape::Data.Shape, params::Dict) = new(shape, NodeQueue(), TreeNode(shape), Params(params))
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
        if node.low != nothing
            visitChildren(node.low)
        end
        if node.low != nothing
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

function splitLeaf(node::TreeNode, onAttr::Int64, epsilon::Float64)

    # Clean out old objects
    node.m = Model.Counting( node.datashape )
    node.p = Model.Probability( node.m )
    node.updatedAttrs = IntSet()
    node.height = 1

    for n in getParentVisitor(node)
        n.height = max(n.low.height, n.high.height) + 1
    end

    node.isLeaf = false
    node.attr = onAttr
    node.epsilon = epsilon
    node.low  = TreeNode(node, node.datashape)
    node.high = TreeNode(node, node.datashape)

    node.low.depth = node.depth + 1
    node.high.depth = node.depth + 1
end

# Not used yet, need to setup pruning
function collapseNode(node::TreeNode)

    if !node.isLeaf

        node.updatedAttrs = IntSet()
        node.m = Model.Counting( node.datashape )

        for n in getVisitor(node)
            Model.merge!(node.m, n.m)
            n.parent = TreeNode()
            n.m = Model.Counting(n.datashape)
            n.p = Model.Probability(n.m)
        end

        node.p = Model.probModel(node.m)
        node.attr=0
        node.isLeaf = true
        node.low = TreeNode()
        node.high = TreeNode()
    end
end

#
#   This is the inner loop of the VFDT where we decide to split the leaf
#       if needed.
#
function updateLeaf(leaf::TreeNode, params::Params, force::Bool)

    leaf.ttl += 1

    pc = 0.0
    for i=1:size(leaf.p.pclass,1)
        if leaf.p.pclass[i] > pc
            pc = leaf.p.pclass[i]
            leaf.label = i
        end
    end

    if leaf.depth < params.maxDepth && leaf.ttl > params.ttl

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

          if gradient > params.delta || gradient < params.tieLimit
              splitLeaf(leaf, bestAttr, epsilon)
          end
        end
    end

end

function updateLeaf(node::TreeNode, params::Params)
    updateLeaf(node::TreeNode, params, false)
end

#
#   Allows us to simply use the tree for clustering...
#
function routeAndCount(stream::Task, tree::HoeffdingTree)

    for l in @task visitLeaves(tree.root)
        l.m = spzeros(size(l.m,1), size(l.m,2))
    end

    for r in stream
        l = sortToLeaf(tree,root, r)
        Model.countRow(r, l.m)
        Model.countFeatureOverlaps(r, l.m)
    end

end

# Training is simple
function train(tree::HoeffdingTree, data::Data.Dataset)
    nbins = tree.shape.unique
    tmpVec = zeros( tree.shape.features * nbins )

    for row in Data.eachrow(data, true)
        fill!(tmpVec,0)
        Model.project!(row, tmpVec, nbins)

        leaf = sortToLeaf(tree.root, tmpVec)
        for t in row.values
            push!(leaf.updatedAttrs, t[1])
        end

        Model.countRow(row, leaf.m)

        updateLeaf(leaf, tree.params)
    end

    return tree
end

function label(tree::HoeffdingTree, params::Dict, stream::Task)

    nbins = tree.shape.unique
    mode = get(params,"mode","naivebayes")
    const naiveBayesLeaves = (mode == "naivebayes")
    const produceRanks = get(params, "ranks", false)

    #   Setup the leaves for prediction
    for leaf in getLeafVisitor(tree.root)
        Model.update(leaf.updatedAttrs, leaf.p, leaf.m)
        if naiveBayesLeaves
          leaf.nb = NaiveBayes.NBPredModel(leaf.p, false)
        end
        leaf.label = indmax(leaf.p.pclass)

        if produceRanks
          rnks = map((i) -> Common.Ranking(i, leaf.p.pclass), 1:size(leaf.p.pclass))
          sort!(rnks, by=(x) -> -x.value)
          leaf.ranks = rnks
        end
    end

    rowAsVec = zeros( tree.shape.features * nbins )

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

export train, label, HoeffdingTree
end # module
