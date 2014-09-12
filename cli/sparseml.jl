#!/usr/bin/env julia
using ArgParse
import SparseML: Common, Data, Model, Measures, VFDT, KNN, NaiveBayes, train, label, save, load
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
        "--confusion"
            help = "Write a confusion matrix to stdout"
            action = :store_true
    end

    return parse_args(s)
end

function writeLabel(label::Int64, labels::IntSet, f::IO)
    real_labels = join(labels," ")
    println(f, label,",",real_labels)
end

function writeLabel(ranks::Vector{Common.Ranking}, labels::IntSet, f::IO)
    real_labels = join(labels," ")
    for r in ranks
        println(f, r.class,":",r.value," ")
    end
    println(f, ",", real_labels)
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
    params = Dict{String,Any}()
    if paramsName != nothing && isfile( paramsName )
        params = JSON.parsefile( paramsName )
    end

    trainingFiles = get(flags, "train", [])
    if length( trainingFiles ) > 0
        println(STDERR,"Started training")
        start = time()
        model = train(model, params, Data.Dataset(trainingFiles))
        delta = time() - start
        println(STDERR," done in $(delta) seconds")
        if flags["save"] != nothing
            save(flags["save"], model)
        end
    end

    testingFiles = get(flags, "label", [])
    if length( testingFiles ) > 0
        println(STDERR,"Started testing")
        start = time()
        testingData = Data.Dataset(testingFiles)
        labelingTask = label(model, params, Data.eachrow(testingData, true))
        outfile = STDOUT
        if flags["pred"] != nothing
            pred_name = flags["pred"]
            outfile = open(pred_name, "w")
            print(STDERR," writing to $(pred_name)")
        else
          print(STDERR," writing to stdout")
        end

        # This doesn't work with ranking...
        if flags["confusion"] == true

          matrix = zeros( testingData.shape.classes, testingData.shape.classes )

          const write_preds = flags["pred"] != nothing

          for lbl in labelingTask
              if write_preds
                writeLabel(lbl[1], lbl[2], outfile)
              end

              for real in lbl[2]
                println(":", lbl)
                matrix[ convert(Int64, lbl[1]) , convert(Int64, real) ] += 1
              end
          end

          println("Confusion matrix")
          for i=1:testingData.shape.classes
            print(STDOUT,matrix[i,1])
            for j=2:testingData.shape.classes
              print(STDOUT,",",matrix[i,j])
            end
            print(STDOUT,"\n")
          end

        else

          for lbl in labelingTask
              writeLabel(lbl[1], lbl[2], outfile)
          end

        end

        delta = time() - start
        println(STDERR," done in $(delta) seconds")

    end

end

main()
