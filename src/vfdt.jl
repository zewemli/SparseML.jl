module VFDT

import ..Common
import ..Data
import ..Model
import ..Measures
import ..NaiveBayes

type TreeNode
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
    
    TreeNode(parent::TreeNode, shape::Data.Shape) = new(true,
                -1, 
                nothing,
                nothing, 
                parent,
                shape,
                Model.Counting(shape),
                nothing,
                nothing,
                0,
                0,
                0,
                0,
                IntSet(),
                nothing)
    
    TreeNode(shape::Data.Shape) = TreeNode(nothing, shape)
end

typealias NodeQueue Collections.PriorityQueue{TreeNode, Float64}

type Params
    ttl::Int64
    delta::Float64
    tieLimit::Float64
    maxDepth::Int64
    G::Function
    
    function Params(params::Dict)
        
        gname = get(params,"measure","gain")
        G = eval(symbol("Measures.$(gname)"))
        
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
    if node.parent != nothing
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
        assert(node.low != nothing)
        assert(node.high != nothing)
        
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
    while p != nothing
        push!(r, p.attr)
        p = p.parent
    end
    return r
end

function splitLeaf(node::TreeNode, onAttr::Int64, epsilon::Float64)
    # Clean out old objects
    node.m = nothing
    node.p = nothing
    node.updatedAttrs = nothing
    node.height = 1

    for n in getParentVisitor(node)
        n.height = max(n.low.height, n.high.height) + 1
    end

    node.isLeaf = false
    node.attr = onAttr
    node.epsilon = epsilon
    node.low  = TreeNode(node.tree, node, node.datashape)
    node.high = TreeNode(node.tree, node, node.datashape)
    
    node.low.depth = node.depth + 1
    node.high.depth = node.depth + 1
end

function collapseNode(node::TreeNode)

    if !node.isLeaf

        node.updatedAttrs = IntSet()
        node.m = Model.Counting( node.datashape )

        for n in getVisitor(node)
            Model.merge!(node.m, n.m)
            n.parent = nothing
            n.m=nothing
            n.p=nothing
        end
        
        node.p = Model.probModel(node.m)
        node.attr=0
        node.isLeaf = true
        node.low=nothing
        node.high=nothing
    end
end

#
#   This is the inner loop of the VFDT where we decide to split the leaf
#       if needed.
#
function updateLeaf(node::TreeNode, params::Params, force::Bool)
    
    leaf.ttl += 1

    if leaf.p == nothing
        leaf.p = Model.probModel( leaf.m )
    else
        Model.update(leaf.updatedAttrs, leaf.p, leaf.m)
    end

    pc = 0.0
    for i=1:size(leaf.p.pclass,1)
        if leaf.p.pclass[i] > pc
            pc = leaf.p.pclass[i]
            leaf.label = i
        end
    end
    
    if leaf.depth < params.maxDepth && leaf.ttl > params.ttl
        ignoreAttrs = prevAttrs(leaf)
        
        # Now loop over attributes and find the best 2 attributes
        aBest = 0
        aNext = 0
        bestG = 0
        nextG = 0
        
        for n=1:length(leaf.p.pfeature)
            nG = params.G(leaf.p, n)
            if nG > bestG
                nextG = bestG
                aNext = aBest
                bestG = nG
                aBest = n
            elseif nG > nextG
                nextG = nG
                aNext = n
            end
        end
        
        epsilon = sqrt( log(1/params.delta) / (2 * leaf.m.n)  )
        gradient = bestG - nextG
        
        if gradient > params.delta || gradient < params.tieLimit
            splitLeaf(leaf, aBest)
        end
        
        node.ttl=0
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

    for row in streams
        Model.project!(row, tmpVec, nbins)
        
        leaf = sortToLeaf(tree.root, tmpVec)
        for t in row.values
            push!(leaf.updatedAttrs, t[1])
        end
        
        Model.countRow(row, leaf.m)
        updateLeaf(leaf, params)
    end
    
    return tree
end

function label(tree::HoeffdingTree, params::Dict, stream::Task)

    mode = get(params,"mode","naivebayes")
    const naiveBayesLeaves = (mode == "naivebayes")
    const produceRanks = get(params, "ranks", false)

    #   Setup the leaves for prediction
    for leaf in getLeafVisitor(tree.root)
        Model.update(leaf.updatedAttrs, leaf.p, leaf.m)
        if naiveBayesLeaves
            leaf.nb = NaiveBayes.NBPredModel(leaf.m)
        end
        leaf.label = indmax(leaf.p.pclass)
        
        if produceRanks
            rnks = map((i) -> Common.Ranking(i, leaf.p.pclass), 1:size(leaf.p.pclass))
            sort!(rnks, by=(x) -> -x.value)
            leaf.ranks = rnks
        end
    end
    
    # Do the actual labelling
    if produceRanks
        for row in stream
            produce( sortToLeaf(tree.root, row).ranks  )
        end
    else
        for row in stream
            produce( sortToLeaf(tree.root, row).label  )
        end
    end

end

export train, label, HoeffdingTree
end # module