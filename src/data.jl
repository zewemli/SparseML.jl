module Data

# needed for the cdf function
using Distributions

typealias Value (Int64,Float64)

type Row
    labels::IntSet
    qid::Int64
    values::Vector{Value}
end

type DataStats
    nClasses::Int64
    nFeatures::Int64
    maxValue::Float64
    minValue::Float64
    density::Float64
    nRows::Int64
    nUnique::Int64
end

function arffParse(line::ASCIIString, classMap::Dict{ASCIIString,Int64})

    row = Row(IntSet(), 0, [])

    if startswith(line,"{")
    
        kstart = 1
        knext = 0
        while kstart > 0
            knext = searchindex(line,",", kstart+1)
            kmid = searchindex(line," ", kstart)
            
            kID = parseint(line[kstart+1 : kmid-1])
            if knext > 0
                kVal  = line[kmid+1 : knext-1]    
            else
                # the end-2 part skips the new line and the final "}"
                kVal = line[kmid+1 : end-2]
            end
            push!(row.values, (kID+1, parsefloat(kVal)))
            kstart = knext
        end
    
    else
    
        classID += 1
        i=1
        k=1
        while k > 0
            kNext = searchindex(line, ",", k+1)
            if i == classID
                push!(row.labels, get!(classMap, line[k+1:kNext-1], length(classMap)+1))
            else
                push!(row.values, (i, parsefloat(line[k+1:kNext-1])))
            end
            i += 1
            k = kNext
        end
    
    end

    return row
end

function svmlightParse(line::ASCIIString, nClasses::Int64)
    row = Row(IntSet(),0,[])
    featStart = rsearchindex(line, " ", searchindex(line, ":"))
    # Get labels
    for l in split(strip(line[1:featStart-1]), " ")
        try
            lInt = parseint(l)
            if lInt != 0
                lInt = lInt < 0 ? nClasses + lInt + 1 : lInt
                push!(row.labels, lInt)
            end
        catch y
            println(y,":",l,"|")
        end
    end
    
    # Check for qid
    if line[featStart+1] == "q"
        qidStart = featStart+1
        featStart = searchindex(line, " ", qidStart)
        row.qid = parseint( line[ searchindex(line,":",qidStart)+1 : featStart ] )
    end
    
    # Get features
    while featStart > 0
        featMid = searchindex(line, ":", featStart)
        featEnd = searchindex(line," ", featMid)
        try
            if featEnd > 0
                push!(row.values, ( parseint(line[featStart:featMid-1]), parsefloat(line[featMid+1:featEnd]) ) )
            else
                push!(row.values, ( parseint(line[featStart:featMid-1]), parsefloat(line[featMid+1:end]) ))
            end
        catch
        
        end
        
        featStart = featEnd
    end
    
    return row
end

function svmlightNext(stream::IOStream, nClasses::Int64)
    line = readline(stream)
    # Skip comments
    while !eof(stream) && ((length(line) == 0) || (line[1] == '#'))
        line = readline(stream)
    end
    
    if length(line)==0
        return Nothing()
    else
        return svmlightParse(line, nClasses)
    end
end

function arffNext(stream::IOStream, classMap::Dict{ASCIIString,Int64})
    line = readline(stream)
    # Skip comments
    while !eof(stream) && ((length(line) == 0) || (line[1] == '%'))
        line = readline(stream)
    end
    
    if length(line)==0
        return Nothing()
    else
        return arffParse(line, classMap)
    end
end

function svmlightIter(stream::IOStream, nClasses::Int64, closeOnEnd::Bool)
    rownumber=1
    ptime = time()
    row = svmlightNext(stream, nClasses)
    while row != nothing
        
        produce( row )
        try
            row = svmlightNext(stream, nClasses)
            rownumber+=1
        catch y
            write(STDERR, string(rownumber,": exception", y,"\n"))    
        end
        
    end
    
    if closeOnEnd
        close(stream)
    end
    
end

# ARFF has not been tested and isn't really supported yet...
function arffIter(stream::IOStream, closeOnEnd::Bool)
    
    # Move past the arff header
    lastNum = -1
    for l in eachline(stream)
        if beginswith(l, "@data") || beginswith(l, "@DATA")
            break
        elseif beginswith(l, "@attribute")
            lastNum += 1
        end
    end
    
    if classID < 0
        classID = lastNum
    end
    
    row = arffNext(stream, classMap, classID)
    while row != nothing
        produce( row )
        row = arffNext(stream, classMap, classID)
    end
    
    if closeOnEnd
        close(stream)
    end
end

function readsparse(stream::IOStream)
    @task svmlightIter(stream, 0, false)
end

function readsparse(stream::IOStream, nClasses::Int64)
    @task svmlightIter(stream, nClasses, false)
end

function readsparse(path::String)
    stream = open(path,"r")
    if endswith(path, ".dat")
        return @task svmlightIter(stream, true)
    elseif endswith(path, ".arff")
        write(STDERR, ".arff not implemented yet...")
    else
        write(STDERR, "Not sure what format ",path," is, I can parse .dat files.")
        quit()
    end
end

function readsparse(stream::IOStream, classMap::Dict{ASCIIString,Int64})
    @task arffIter(stream, classMap)
end

function writesparse(rows::Task, stream::IOStream)

    for row in rows
        write(stream, join(row.labels, " "))
        if row.qid != 0
            q = row.qid
            write(stream, " qid:$q")
        end
        
        for (i,v) in row.values
            write(stream, string(" ",i,":",convert(Int64, v)))
        end
        write(stream, "\n")
    end

end

function getStats(stream::Task)
    classes = IntSet()
    features = IntSet()
    vals = Set{Float64}()
    minVal = 0 # <- 0 is the missing value, so just assume it was in there...
    maxVal = -Inf
    nvals = 0
    nrows = 0
    for row in stream
        nrows += 1
        union!(classes, row.labels)
        for (i,v) in row.values
            push!(features, i)
            push!(vals, v)
            minVal = min(minVal, v)
            maxVal = max(maxVal, v)
            nvals += 1
        end
    end
    
    DataStats(length(classes), maximum(features), maxVal, minVal, nvals/(maximum(features) * nrows), nrows, length(vals))
end

function getStats(stream::IOStream)
    seekstart(stream)
    stats = getStats(readsparse(stream))
    seekstart(stream)
    return stats
end

function getStats(path::ASCIIString)
    f = open(path,"r")
    stats = getStats(f)
    close(f)
    return stats
end

# Equal width bins
function discretizer(stream::Task, limits::Array{Float64,2}, bins::Int64)

    mult = [ (1/(limits[i,2] - limits[i,1]))*bins for i=1:size(limits,1) ]

    for r in stream
        vals = Value[]
        
        for v in r.values
            lv = clamp(floor(( v[2] - limits[v[1],1] ) * mult[ v[1] ])+1, 1, bins)
            kvalue = (v[1]-1)*bins + lv
            
            if isfinite(kvalue)
                push!(vals, (convert(Int64, kvalue), 1,))
            else
                push!(vals, (convert(Int64, (v[1]-1)*bins + 1), 1,))
            end
        end
        
        produce( Row(r.labels, r.qid, vals) ) 
    end

end

# Threshold with cdf function
function discretizer(stream::Task, dist::Vector{Any}, bins::Int64)
    
    for r in stream
        vals = Value[]
        
        for v in r.values
            dv = dist[ v[1] ]
            lv = 1
            if typeof(dv) == Dict{Float64,Int64}
                # All unobserved values get put into bin 0
                lv = get!(dv, v[2], 0)
            else
                lv = clamp(floor(cdf(dv, v[2]) * bins) + 1, 1, bins)
            end
            push!(vals, (convert(Int64, (v[1]-1)*bins + lv), 1))
        end
        
        produce( Row(r.labels, r.qid, vals) ) 
    end
end

export discretizer, readsparse, writesparse, Row, Value, DataStats, getStats
end # module
