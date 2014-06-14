#!/usr/bin/env julia
using ArgParse
import SparseML: Common, Data, Model, Measures, VFDT, KNN, NaiveBayes
using JSON

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--model", "-m"
            help = "Name of the algorithm to use, or path to saved model, Algorithms are [knn, naivebayes, vfdt]"
        "--label", "-l"
            help = "Files to label"
            nargs = '*'
        "--train", "-t"
            help = "Training files"
            nargs = '*'
        "--params", "-p"
            help = "Path to JSON parameters file"
        "--save"
            help = "Path to which the trained model should be saved"
        "--pred"
            help = "Save predictions to this file"
    end

    return parse_args(s)
end

function writeLabel(label::Int64, f::IOStream)
    write(f, "$(label)\n")
end

function writeLabel(ranks::Vector{Common.Ranking}, f::IOStream)
    for r in ranks
        write(f, "$(r.class):$(r.value) ")
    end
    write(f, "\n")
end

function main()
    flags = parse_commandline()
    
    canLabel = false
    model = nothing
    modelName = get(flags, "model", "naivebayes")
    
    if isfile(modelName)
        # Load the model from a file
        model = modelName
    elseif modelName == "naivebayes"
        model = NaiveBayes.NB
    elseif modelName == "knn"
        model = KNN.Subset
    else
        model = VFDT.HoeffdingTree
    end
    
    paramsName = get(flags,"params","")
    params = Dict()
    if isfile( paramsName )
        params = JSON.parsefile( paramsName )
    end
    
    trainingFiles = get(flags, "train", [])
    if length( trainingFiles ) > 0
        model = train(model, params, Data.Dataset(trainingFiles))
        
        if flags["save"] != nothing
            save(flags["save"], model)
        end
    end
    
    testingFiles = get(flags, "test", [])
    if length( testingFiles ) > 0
        testingData = Data.Dataset(testingFiles)
        labelingTask = label(model, params, Data.eachrow(testingData))
        
        outfile = STDOUT
        if flags["pred"] != nothing
            outfile = open(flags["pred"], "w")
        end
        
        for lbl in labelingTask
            writeLabel(lbl, outfile)
        end
    end
    
end
main()