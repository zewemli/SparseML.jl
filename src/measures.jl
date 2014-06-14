module Measures

import ..Data
import ..Model

type ConfusionMatrix
    tp::Int64
    fp::Int64
    tn::Int64
    fn::Int64
    ConfusionMatrix() = new(0,0,0,0)
end

function inc(m::ConfusionMatrix, prediction::Bool, reality::Bool)
    if prediction
        if reality
            m.tp += 1
        else
            m.fp += 1
        end
    else
        if reality
            m.fn += 1
        else
            m.tn += 1
        end
    end
end

function precision(m::ConfusionMatrix)
    v = m.tp / (m.tp + m.fp)
    isfinite(v) ? v : 0.0
end

function recall(m::ConfusionMatrix)
    m.tp / (m.tp + m.fn)
end

function sensitivity(m::ConfusionMatrix)
    recall(m)
end

function specificity(m::ConfusionMatrix)
    m.tn / (m.fp + m.tn)
end

function npv(m::ConfusionMatrix)
    m.tn / (m.tn + m.fn)
end

function fallout(m::ConfusionMatrix)
    m.fp / (m.fp + m.tn)
end

function falsediscovery(m::ConfusionMatrix)
    m.fp / (m.tp + m.fp)
end

function missrate(m::ConfusionMatrix)
    m.fn / (m.fn + m.tp)
end

function accuracy(m::ConfusionMatrix)
    (m.tp + m.tn) / (m.tp + m.tn + m.fp + m.fn)
end

function fbeta(m::ConfusionMatrix, beta::Float64)
    (1 + (beta ^ 2)) * ((precision(m) * recall(m)) / (((beta^2) * precision(m)) + recall(m)))
end

function f1(m::ConfusionMatrix)
    fbeta(m, 1.0)
end

function mcc(m::ConfusionMatrix)
    try
        return ((m.tp * m.tn) - (m.fp * m.fn)) / sqrt( (m.tp + m.fp) * (m.tp + m.fn) * (m.tn + m.fp) * (m.tn + m.fn) )
    catch
        return 0.0
    end
end

function informedness(m::ConfusionMatrix)
    sensitivity(m) + specificity(m) - 1
end

function markedness(m::ConfusionMatrix)
    precision(m) + npv(m) - 1
end

function macrof1(M::Vector{ConfusionMatrix})
    p = sum([ precision(M[i]) for i=1:length(M) ]) / length(M)
    r = sum([ recall(M[i]) for i=1:length(M) ]) / length(M)
    (2 * p * r) / (p+r)
end

# Using L1
function dist(r1::Data.Row, r2::Data.Row)

  d = 0.0

  const len1 = size(r1.values,1)
  const len2 = size(r2.values,1)

  i=1
  j=1

  # This looks a bit cryptic, but we a just walking
  # through the values, which are sorted by index,
  # in order to measure distance

  while i <= len1 && j <= len2

    t1 = r1.values[i][1]
    t2 = r2.values[j][1]

    feat1 = t1[1]
    feat2 = t2[1]

    if feat1 == feat2
      d += t1[2] - t2[2]
    elseif feat1 < feat2
      while i <= len1 && r1.values[i][1] < feat2
        j+=1
      end
    else # feat2 < feat1
      while j <= len2 && r2.values[j][1] < feat1
        j+=1
      end
    end

  end

  # One may still have more values
  if i < len1
    while i <= len1
      d += r1.values[i][2]
    end
  else j < len2
    while j <= len2
      d += r2.values[j][2]
    end
  end

  return d
end

function dist(r1::Data.Row, r2::Data.Row, k::Float64)

  d = 0.0

  const len1 = size(r1.values,1)
  const len2 = size(r2.values,1)

  i=1
  j=1

  # This looks a bit cryptic, but we a just walking
  # through the values, which are sorted by index,
  # in order to measure distance

  while i <= len1 && j <= len2

    t1 = r1.values[i][1]
    t2 = r2.values[j][1]

    feat1 = t1[1]
    feat2 = t2[1]

    if feat1 == feat2
      d += (t1[2] - t2[2]) ^ k
    elseif feat1 < feat2
      while i <= len1 && r1.values[i][1] < feat2
        j+=1
      end
    else # feat2 < feat1
      while j <= len2 && r2.values[j][1] < feat1
        j+=1
      end
    end

  end

  # One may still have more values
  if i < len1
    while i <= len1
      d += r1.values[i][2]
    end
  else j < len2
    while j <= len2
      d += r2.values[j][2]
    end
  end

  return d^(1/k)
end

function dist(v1::Vector{Float64}, v2::Vector{Float64})
  return sum(v1-v2)
end

function dist(v1::Vector{Float64}, v2::Vector{Float64}, k::Float64)
  return sum( .^(.-(v1,v2),k) )^(1/k)
end

function entropy(V, pMin)
    e = 0.0
    low = pMin * log(pMin)
    for p in V
        e -= p == 0 ? low : p*log(p)
    end
    return e
end

function kl(m::Model.Probability, byClass::Bool)

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

function npmi(m::Model.Probability, byClass::Bool)
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

function gini(m::Model.Probability, attr::Int64)
    s = 0.0
    for i=1:size(m.prob,1)
        s += m.prob[i,attr]^2
    end
    return 1-s
end

function gain(m::Model.Probability, attr::Int64)
    s = 0.0
    for i=1:size(m.prob,1)
        v = m.prob[i,attr]
        s += v*log(v)
    end
    return -s
end

export ConfusionMatrix, 
        dist,
        inc,
        precision,
        recall,
        sensitivity,
        specificity,
        npv, 
        fallout, 
        falsediscovery, 
        missrate, 
        accuracy,
        f1, 
        fbeta,
        mcc,
        informedness,
        markedness,
        macrof1, 
        kl, 
        npmi, 
        gini,
        gain
end # module
