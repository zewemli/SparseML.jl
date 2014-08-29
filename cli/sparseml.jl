#!/usr/bin/env julia
using ArgParse
import SparseML: Common, Data, Model, Measures, VFDT, KNN, NaiveBayes, train, label
using JSON

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--model", "-m"
            help = "Name of the algorithm to use, or path to saved model, Algorithms are [knn, naivebayes, vfdt]"
            arg_type = String
        "--label", "-l"
            help = "Files to label"
            nargs = '*'
            arg_type = String
        "--train", "-t"
            help = "Training files"
            nargs = '*'
            arg_type = String
        "--params", "-p"
            help = "Path to JSON parameters file"
            arg_type = String
        "--save"
            help = "Path to which the trained model should be saved"
            arg_type = String
        "--pred"
            help = "Save predictions to this file"
            arg_type = String
    end

    return parse_args(s)
end

function writeLabel(label::Int64, labels::IntSet, f::IOStream)
    real_labels = join(labels," ")
    write(f, "$(label),$(real_labels)\n")
end

function writeLabel(ranks::Vector{Common.Ranking}, labels::IntSet, f::IOStream)
    real_labels = join(labels," ")
    for r in ranks
        write(f, "$(r.class):$(r.value) ")
    end
    write(f, ",$(real_labels)\n")
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

    paramsName = get(flags,"params",nothing)
    params = Dict()
    if paramsName != nothing && isfile( paramsName )
        params = JSON.parsefile( paramsName )
    end

    trainingFiles = get(flags, "train", [])
    if length( trainingFiles ) > 0
        println("Started training")
        start = time()
        model = train(model, params, Data.Dataset(trainingFiles))
        delta = time() - start
        println(" done in $(delta) seconds")
        if flags["save"] != nothing
            save(flags["save"], model)
        end
    end

    testingFiles = get(flags, "label", [])
    if length( testingFiles ) > 0
        println("Started testing")
        start = time()
        testingData = Data.Dataset(testingFiles)
        labelingTask = label(model, params, Data.eachrow(testingData, true))
        outfile = STDOUT
        if flags["pred"] != nothing
            pred_name = flags["pred"]
            outfile = open(pred_name, "w")
            print(" writing to $(pred_name)")
        else
          print(" writing to stdout")
        end

        for lbl in labelingTask
            writeLabel(lbl[1], lbl[2], outfile)
        end

        delta = time() - start
        println(" done in $(delta) seconds")

    end

end

main()
