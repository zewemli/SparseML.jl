module SparseML
#
#   Import algorithms here and bundle together algorithms
#   which have the same requirements
#
include("common.jl")

include("data.jl")
include("model.jl")
include("measures.jl")

include("naivebayes.jl")
include("knn.jl")
include("vfdt.jl")

MLModel = Union(Type{VFDT.HoeffdingTree}, Type{NaiveBayes.NB}, Type{KNN.Subset})
MLModelInst = Union(VFDT.HoeffdingTree, NaiveBayes.NB, KNN.Subset)

function load(path::String)
  open(path) do f
    return deserialize(f)
  end
end

function save(path::String, knn::KNN.Subset)
  Data.write(knn.shape, knn.subset, path)
end

function save(path::String, model::MLModelInst)
  open(path, "w") do f
    serialize(f, model)
  end
end

#
#   To train a model we need a model type and labelled data
#
function train(model::String, params::Common.Params, data::Data.Dataset)
  train(load(model), data)
end

function train(modelType::MLModel, params::Common.Params, data::Data.Dataset)
  train(modelType(data.shape, params), data)
end

function label(model::String, params::Common.Params, stream::Task)
  label(load(model), params, stream)
end

# VFDT
function train(model::VFDT.HoeffdingTree, data::Data.Dataset)
  VFDT.train(model, data)
end

function label(model::VFDT.HoeffdingTree, params::Common.Params, stream::Task)
  @task VFDT.label(model, params, stream)
end

# NaiveBayes
function train(model::NaiveBayes.NB, data::Data.Dataset)
  NaiveBayes.train(model, data)
end

function label(model::NaiveBayes.NB, params::Common.Params, stream::Task)
  @task NaiveBayes.label(model, params, stream)
end

# K-Nearest Neighbors
function train(model::KNN.Subset, data::Data.Dataset)
  KNN.train(model, data)
end

function label(model::KNN.Subset, params::Common.Params, stream::Task)
  @task KNN.label(model, params, stream)
end

export train, label, load, save
end # module
