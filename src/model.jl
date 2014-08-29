module Model

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

    Counting(shape::Data.Shape) = Counting(shape.classes, shape.features, shape.unique)

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

    function Probability(m::Counting)
        extraN = estExtra(m)
        pMin = estPMin(m, extraN)

        prob = new( spzeros( size(m.overlap,1), size(m.overlap, 2) ),
                    spzeros( size(m.featureoverlap,1), size(m.featureoverlap,2) ),
                    spzeros( size(m.classoverlap,1), size(m.classoverlap,2) ),
                    zeros( size(m.classcount) ),
                    zeros( size(m.featurecount) ),
                    m.nBins,
                    pMin )

        if m.n > 0
            I,J,V = findnz(m.overlap)

            prob.pclass = ./(m.classcount, m.n)
            prob.pfeature = ./(m.featurecount, m.n)

            for (c, f, f_and_c) in zip(I,J,V)
              cf_val = f_and_c / (m.classcount[c] + extraN*prob.pclass[c])
              prob.prob[ c,f ] = isfinite(cf_val) ? cf_val : 0
            end

            I,J,V = findnz(m.featureoverlap)
            for (i,j,v) in zip(I,J,V)
              ij_val = v / (m.featurecount[i] + m.featurecount[j])
              prob.poverlap[ i,j ] = isfinite(ij_val) ? ij_val : 0
            end

            I,J,V = findnz(m.classoverlap)
            for (i,j,v) in zip(I,J,V)
              ijv = v / (m.classcount[i] + m.classcount[j])
              prob.pclassoverlap[ i,j ] = isfinite(ijv) ? ijv : 0
            end

        end

        return prob
    end

end

# Estimate the number of examples we would need
# to achieve full density
function estExtra( model::Counting )
    log(2, model.n) * (length(model.overlap) - nnz(model.overlap))
end

function estPMin( model::Counting, extra::Float64 )
    1 / (model.n + extra)
end

function update( attrs::IntSet, pmodel::Probability, model::Counting )
    nclasses = length(model.classcount)
    nfeatures = length(model.featurecount)

    pmodel.pclass = ./(model.classcount, model.n)
    pmodel.pfeature = ./(model.featurecount, model.n)

    checkOverlap = nnz(model.featureoverlap) > 0
    extraN = estExtra( model )

    for a in attrs

        pmodel.pfeature[a] = model.featurecount[a] / model.n
        for c=1:nclasses
            ca = model.overlap[c,a] / (model.classcount[c] + extraN*pmodel.pclass[c])
            pmodel.prob[ c, a ] = isfinite(ca) ? ca : 0
        end

        for a2=1:nfeatures
            overlap = model.featureoverlap[a,a2]
            if a != a2 && overlap > 0
                newProb = overlap / (model.featurecount[a] + model.featurecount[a2])
                if a < a2
                    pmodel.poverlap[ a,a2 ] = isfinite(newProb) ? newProb : 0
                else
                    pmodel.poverlap[ a2,a ] = isfinite(newProb) ? newProb : 0
                end
            end
        end
    end

    I,J,V = findnz(model.classoverlap)
    for (i,j,v) in zip(I,J,V)
        ij = v / (model.classcount[i] + model.classcount[j])
        pmodel.pclassoverlap[ i,j ] = isfinite(ij) ? ij : 0
    end
end

function project(sv::Data.Value, nBins::Int64)
    return (sv[1]*nBins) + convert(Int64, sv[2])
end

function project(sv::Data.Value, row::Data.Row)
    project(sv, row.nbins)
end

function project!(r::Data.Row, v::Vector{Float64}, nBins::Int64)
    for val in r.values
        v[ project(val, r) ] = 1
    end
end

# Counting model methods
function countRow(row::Data.Row, model::Counting)

    nclasses = size(model.classcount,1)

    model.n += 1

    for t in row.values
        model.featurecount[ project(t, row) ] += 1
    end

    for c in row.labels

        model.classcount[c] += 1

        for t in row.values
            model.overlap[ c, project(t, row) ] += 1
        end

    end
end

function countFeatureOverlaps(row::Data.Row, model::Counting)
    n = length(row.values)
    for i=1:n
        iVal = project(row.values[i], model.nBins)
        for j=i+1:n
            model.featureoverlap[ iVal, project(row.values[j], row) ] += 1
        end
    end
end

function count(data::Data.Dataset, model::Counting)
    count(Data.eachrow(data), model)
end

function count(rows::Task, model::Counting)
    count(rows,model,false)
end

function count(rows::Task, model::Counting, includeFeatureOverlaps::Bool)

    if includeFeatureOverlaps
        for row in rows
            countRow(row, model)
            countFeatureOverlaps(row, model)
        end
    else
        for row in rows
            countRow(row, model)
        end
    end
    return model
end

function count(rows::Task, nClasses::Int64, nFeatures::Int64, maxValsPerFeat::Int64)
    count(rows, nClasses, nFeatures, maxValsPerFeat, false)
end

function count(rows::Task, nClasses::Int64, nFeatures::Int64, maxValsPerFeat::Int64, includeFeatureOverlaps::Bool)
    count(rows, Counting(nClasses, nFeatures, maxValsPerFeat), includeFeatureOverlaps)
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

export count, merge!, project, project!
end # module
