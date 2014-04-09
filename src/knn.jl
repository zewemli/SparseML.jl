module KNN

include("data.jl")
include("model.jl")

function predictIter(rows::Task, model::Model.Probability, ignoreClass::Bool, n::Int64, p::Float64)
end


function predict(rows::Task, model::Model.Counting, ignoreClass::Bool, n::Int64, p::Float64)
    @task predictIter(rows, countsToPredModel(model), ignoreClass, min(n, length(model.classcount)), p)
end

function predict(rows::Task, model::Model.Counting, ignoreClass::Bool, n::Int64)
    @task predictIter(rows, countsToPredModel(model), ignoreClass, min(n, length(model.classcount)), 0)
end

function predict(rows::Task, model::Model.Counting, ignoreClass::Bool, p::Float64)
    @task predictIter(rows, countsToPredModel(model), ignoreClass, 0, p)
end

function predWriter(predTask::Task, output::IOStream, n::Int64)
    if n == 1
        return @task predWriteSingle(predTask, output)
    else
        return @task predWriteMany(predTask, output)
    end
end

function predWriteMany(predTask::Task, output::IOStream)
    
    for pk in predTask
        reality = pk[1]
        prediction = pk[2]
        write(output, string(join(reality," "),",",join([ string(k[2],"|",k[1]) for k in prediction ], " "),"\n"))
        produce(pk)
    end

end

function predWriteSingle(predTask::Task, output::IOStream)

    # prediction is a tuple (prob, class)
    for px in predTask
        reality = px[1]
        prediction = px[2][2]
        predictionProb = px[2][1]
        write(output, string(join(reality," "),",",prediction,",",predictionProb,"\n"))
        produce(px)
    end

end

export predict, predWriter
end # module
