module NaiveBayes

import ..Model

function setProb!(features::Vector{Int64}, conditional::SparseMatrixCSC{Float64,Int64}, pMin::Float64, probs::Vector{Float64})
    for c=1:size(probs,1)
        for j in features
            cj = conditional[c,j]
            probs[c] += cj == 0 ? pMin : cj
        end
    end
    return probs
end

function predictIter(rows::Task, model::Model.Probability, ignoreClass::Bool, n::Int64, p::Float64)

    conditional = spzeros( size(model.prob,1), size(model.prob,2) ) 
    rowProb = zeros( size(model.pclass) )
    priors = ignoreClass ? zeros( size(model.pclass) ) : map(log, model.pclass)
    pMin = log(model.pMin)
    pEmpty = zeros( size(model.pclass) )
    
    I,J,V = findnz(model.prob)
    
    for (c, f, p_f_given_c) in zip(I,J,V)
        p_neg = log(1 - p_f_given_c)
        pEmpty[c] += p_neg
        conditional[c,f] = log(p_f_given_c) - p_neg
    end
    
    # add in pEmpty to precompute if all features == 0
    priors = .+(priors, pEmpty)
    
    start = time()
    i = 0
    rownumber=0
    
    for row in rows
        rownumber+=1
        i+=1
        
        # Initialize the probabilities
        rowProb[1:end] = priors
        
        features = [ Model.project(V, model.nBins) for V in row.values ]
        
        setProb!(features, conditional, pMin, rowProb)
        
        if p > 0 || n > 1
        
            czipped = collect(zip(./(rowProb,abs(maximum(rowProb))), 1:length(rowProb)))
            
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
            for c=1:length(rowProb)
                if rowProb[c] > pMax
                    pMax = rowProb[c]
                    cMax = c
                end
            end
            produce( (row.labels, (pMax, cMax)) )
        end
        
        if i == 1000
            write(STDERR, string("predicted ", rownumber, " @ ~", i / (time() - start), " lps\n"))
            i=0; start=time()
        end
    end
    
end

function predict(rows::Task, model::Model.Counting, ignoreClass::Bool, n::Int64, p::Float64)
    @task predictIter(rows, Model.probModel(model), ignoreClass, min(n, length(model.classcount)), p)
end

function predict(rows::Task, model::Model.Counting, ignoreClass::Bool, n::Int64)
    @task predictIter(rows, Model.probModel(model), ignoreClass, min(n, length(model.classcount)), 0)
end

function predict(rows::Task, model::Model.Counting, ignoreClass::Bool, p::Float64)
    @task predictIter(rows, Model.probModel(model), ignoreClass, 0, p)
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