module Model
using Distributions

import ..Data

# Provide a model for counting overlaps and intersections
# this is the model which should be saved to disk
type Counting
    overlap::SparseMatrixCSC{Float64,Int64}
    classcount::Vector{Int64}
    featurecount::Vector{Int64}
    classoverlap::SparseMatrixCSC{Float64,Int64}
    featureoverlap::SparseMatrixCSC{Float64,Int64}

    nBins::Int64
    n::Int64

    Counting(nClasses::Int64, nFeatures::Int64, maxValsPerFeat::Int64) = new(spzeros(nClasses, (nFeatures+1)*maxValsPerFeat),
                                        zeros(nClasses),
                                        zeros( (nFeatures+1)*maxValsPerFeat ),
                                        spzeros(nClasses, nClasses),
                                        spzeros(maxValsPerFeat*(nFeatures+1), maxValsPerFeat*(nFeatures+1)),
                                        maxValsPerFeat,
                                        0)
                                        
    Counting() = new(spzeros(0,0), Int64[], Int64[], spzeros(0,0), spzeros(0,0), 0, 0)
end

type Probability

    prob::SparseMatrixCSC{Float64,Int64}
    poverlap::SparseMatrixCSC{Float64,Int64}
    pclassoverlap::SparseMatrixCSC{Float64,Int64}
    pclass::Vector{Float64}
    pfeature::Vector{Float64}
    nBins::Int64
    pMin::Float64
    Probability(m::Counting, pMin::Float64) = new( 
                                            spzeros( size(m.overlap,1), size(m.overlap, 2) ),
                                            spzeros( size(m.featureoverlap,1), size(m.featureoverlap,2) ),
                                            spzeros( size(m.classoverlap,1), size(m.classoverlap,2) ),
                                            zeros( size(m.classcount) ),
                                            zeros( size(m.featurecount) ),
                                            m.nBins,
                                            pMin)
end

type Continuous

    dist::Array{Distribution,2}
    pclass::Vector{Float64}
    
end

PredModel = Union(Probability, Continuous)

function prior(m::PredModel, c)
    return m.pclass[c]
end

function conditional(m::Continuous, c::Int64, f::Data.Value)
    logcdf(m.dist[c, f[1]], f[2])
end

function conditional(m::Probability, c::Int64, f::Data.Value)
    log(m.prob[c, project(f, m.nBins)])
end

function conditional(m::SparseMatrixCSC{Float64,Int64}, c::Int64, f::Data.Value)
    m[c, project(f, m.nBins)]
end

function probModel( model::Counting )

    I,J,V = findnz(model.overlap)

    nMissing = length(model.overlap) - nfilled(model.overlap)
    # extra examples required to get full density
    extraN = nMissing * log(2, model.n)

    scale = 1 / (model.n + extraN)
    pMin = scale

    pmodel = Probability(model, pMin)
    pmodel.pclass = ./(model.classcount, model.n)
    pmodel.pfeature = ./(model.featurecount, model.n)
    
    for (c, f, f_and_c) in zip(I,J,V)
        pmodel.prob[ c,f ] = f_and_c / (model.classcount[c] + extraN*pmodel.pclass[c])
    end
    
    I,J,V = findnz(model.featureoverlap)
    for (i,j,v) in zip(I,J,V)
        pmodel.poverlap[ i,j ] = v / (model.featurecount[i] + model.featurecount[j])
    end
    
    I,J,V = findnz(model.classoverlap)
    for (i,j,v) in zip(I,J,V)
        pmodel.pclassoverlap[ i,j ] = v / (model.classcount[i] + model.classcount[j])
    end
    
    return pmodel
end

function project(sv::Data.Value, nBins::Int64)
    return (sv[1]*nBins) + convert(Int64, sv[2])
end

# Counting model methods
function count_row(row::Data.Row, model::Counting)

    nclasses = size(model.classcount,1)
    
    model.n += 1

    for t in row.values
        model.featurecount[ project(t, model.nBins) ] += 1
    end

    for c in row.labels
        
        model.classcount[c] += 1
    
        for t in row.values
            model.overlap[ c, project(t, model.nBins) ] += 1
        end
    
    end
end

function count_feature_overlaps(row::Data.Row, model::Counting)
    n = length(row.values)
    for i=1:n
        iVal = project(row.values[i], model.nBins)
        for j=i+1:n
            model.featureoverlap[ iVal, project(row.values[j], model.nBins) ] += 1
        end
    end
end

function count(rows::Task, model::Counting)
    count(rows,model,false)
end

function count(rows::Task, model::Counting, include_feature_overlaps::Bool)
    
    if include_feature_overlaps
        for row in rows
            count_row(row, model)
            count_feature_overlaps(row, model)
        end
    else
        for row in rows
            count_row(row, model)
        end
    end
    return model
end

function count(rows::Task, nClasses::Int64, nFeatures::Int64, maxValsPerFeat::Int64)
    count(rows, nClasses, nFeatures, maxValsPerFeat, false)
end

function count(rows::Task, nClasses::Int64, nFeatures::Int64, maxValsPerFeat::Int64, include_feature_overlaps::Bool)
    count(rows, Counting(nClasses, nFeatures, maxValsPerFeat), include_feature_overlaps)
end

function merge!(dest::Counting, src::Counting)
    
    for i=1:length(src.classcount)
        dest.classcount[i] += src.classcount[i]
    end
    
    for i=1:length(src.featurecount)
        dest.featurecount[i] += src.featurecount[i]
    end
    
    I,J,V = findnz(src.overlap)
    for (i,j,v) in zip(I,J,V)
        dest.overlap[i,j] += v
    end

    I,J,V = findnz(src.classoverlap)
    for (i,j,v) in zip(I,J,V)
        dest.classoverlap[i,j] += v
    end
    
    I,J,V = findnz(src.featureoverlap)
    for (i,j,v) in zip(I,J,V)
        dest.featureoverlap[i,j] += v
    end
    
    dest.n += src.n
    
    return dest
end

function fit(rows::Task, F::Distribution, nClasses::Int64, nFeatures::Int64)

    countClass = zeros(nClasses)
    cvals = [ [ Float64[] for j=1:nFeatures ] for i=1:nClasses ]
    
    n=0
    for r in rows
        n += 1
        for c in r.labels
            countClass[c] += 1
            for v in r.values
                push!(cvlas[c][ v[1] ], v[2])
            end
        end
    end
    
    model = Continuous( Array(Distribution, (nClasses, nFeatures)), zeros(nClasses) )
    
    # Set the class priors
    for i=1:nClasses
        model.pclass[i] = classCount[i] / n
    
        # Now fit the features
        for j=1:nFeatures
            model.dist[i,j] = fit_mle(F, cvals[i,j])
        end
    end
    
    return model
    
end

function entropy(V, pMin)
    e = 0.0
    low = pMin * log(pMin)
    for p in V
        e -= p == 0 ? low : p*log(p)
    end
    return e
end

function kl(m::Probability, byClass::Bool)
    
    M = byClass ? m.prob : m.poverlap
    nrow = size(M,1)
    ncol = size(M,2)
    target = zeros( (nrow,nrow) )

    if m.pAreLogs
        write(STDERR, "Model.kl needs to raw probabilities instead of the log probabilites.\n")
    
    else
    
        for p=1:nrow
            for q=1:nrow
                ij_kl = 0.0
                
                for c=1:ncol
                    pc = max(m.pMin, M[p,c]) 
                    ij_kl += log(pc / max(m.pMin, M[q,c])) * pc 
                end
                
                target[i,j] = ij_kl
            end
        end
    
    end
    
    return target
end

function npmi(m::Probability, byClass::Bool)

    M = byClass ? m.prob : m.poverlap
    P_i = byClass ? m.pclass : m.pfeature
    P_j = m.pfeature
    nrow = size(M,1)
    ncol = size(M,2)
    target = spzeros( nrow, nrow )

    for i=1:nrow
        pi = max(P_i[i], m.pMin)
        for j=i+1:nrow
            pj = max(P_j[j], m.pMin)
            p_ij = max(m.pMin, M[i,j])
            target[i,j] = ( p_ij / (pi*pj) ) / -log(p_ij)
            target[j,i] = target[i,j]
        end
    end
    
    return target
    
end

export probModel, kl, npmi, count, merge!, project, PredModel
end # module
