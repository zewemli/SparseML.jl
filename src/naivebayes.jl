module NaiveBayes

include("data.jl")
include("model.jl")

function setProb!(features::Vector{Int64}, conditional::SparseMatrixCSC{Float64,Int64}, probs::Vector{Float64})
    for c=1:length(m.pclass)
        for j in features
            probs[c] += conditional[c,j]
        end
    end
    return probs
end

function predictIter(rows::Task, model::Model.Probability, ignoreClass::Bool, n::Int64, p::Float64)

    conditionalGain = spzeros( size(model.prob,1), size(model.prob,2) ) 
    priors = ignoreClass ? zeros( size(model.pclass) ) : map(log, model.pclass) 
    rowprob = zeros( size(model.pclass) )
    
    pEmpty = zeros( size(model.pclass) )
    pNotFeat = map(log, .-(1, model.pfeature) )
    featGain = [ log(v) - log(1-v) for v in model.pfeature ]
    pNoFeats = sum(pNotFeat)
    
    I,J,V = findnz(model.prob)
    for (i,j,v) in zip(I,J,V)
        p_ij = v / model.n
        p_neg = log(1 - p_ij)
        pEmpty[i] += p_neg
        conditionalGain[i,j] = log(p_ij) - p_neg
    end
    
    priors = .+(priors, pEmpty)
    pNone = sum( pNotFeat )

    start = time()
    i = 0
    rownumber=0
    
    for row in rows
        rownumber+=1
        i+=1
        
        # Initialize the probabilities
        rowprob[1:end] = priors
        features = [ project(V, model.nBins) for V in row.values ]
        
        pdata = pNoFeats
        for f in features
            pdata += featGain[f]
        end
        
        setProb!(row.values, conditionalGain, rowprob)
        
        for i=1:length(pdata)
            rowprob[i] -= pdata[i]
            assert(0 <= rowprob[i] <= 1)
        end
        
        if p > 0 || n > 1
        
            #czipped = collect(zip( ./(prob, sum(prob)) , model.classID))
            czipped = collect(zip(rowprob, model.classID))
            
            sort!(czipped, rev=true)
            if 0 < p < 1
                psum = 0.0
                for c=1:length(czipped)
                    psum += cprob[c][1]
                    if psum >= p
                        break
                    end
                end
                produce((row.labels, czipped[1:c]))
            elseif 0 < n < length(czipped)
                produce((row.labels, czipped[1:n]))
            else
                produce((row.labels, czipped))
            end
            
        else
            pMax = -Inf
            cMax = 0
            for c=1:length(prob)
                if rowprob[c] > pMax
                    pMax = rowprob[c]
                    cMax = c
                end
            end
            produce( (row.labels, (pMax, model.classID[cMax])) )
        end
        
        if i == 1000
            write(STDERR, string("predicted ", rownumber, " @ ~", i / (time() - start), " lps\n"))
            i=0; start=time()
        end
    end
    
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