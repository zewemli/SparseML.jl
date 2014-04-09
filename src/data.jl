module Data

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

function arffParse(line::ASCIIString, classMap::Dict{ASCIIString,Int64}, classID::Int64)

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
            
            if kID == classID
                push!(row.labels, get!(classMap, kVal, length(classMap)+1))
            else
                push!(row.values, (kID+1, parsefloat(kVal)))
            end
            
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

function svmlightParse(line::ASCIIString)
    row = Row(IntSet(),0,[])
    featStart = rsearchindex(line, " ", searchindex(line, ":"))
    # Get features
    for l in split(line[1:featStart-1], ",")
        try
            lInt = parseint(l)
            if lInt != 0
                push!(row.labels, lInt)
            end
        catch
            
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

function svmlightNext(stream::IOStream)
    line = readline(stream)
    # Skip comments
    while !eof(stream) && ((length(line) == 0) || (line[1] == '#'))
        line = readline(stream)
    end
    
    if length(line)==0
        return Nothing()
    else
        return svmlightParse(line)
    end
end

function arffNext(stream::IOStream, classMap::Dict{ASCIIString,Int64}, classID::Int64)
    line = readline(stream)
    # Skip comments
    while !eof(stream) && ((length(line) == 0) || (line[1] == '%'))
        line = readline(stream)
    end
    
    if length(line)==0
        return Nothing()
    else
        return arffParse(line, classMap, classID)
    end
end

function svmlightIter(stream::IOStream, closeOnEnd::Bool)
    rownumber=1
    ptime = time()
    row = svmlightNext(stream)
    while row != nothing
        
        produce( row )
        try
            row = svmlightNext(stream)
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
    @task svmlightIter(stream, false)
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

function readsparse(stream::IOStream, classMap::Dict{ASCIIString,Int64}, classID::Int64)
    @task arffIter(stream, classMap, classID)
end

function getStats(stream::Task)
    classes = IntSet()
    features = IntSet()
    vals = Set{Float64}()
    minVal = Inf
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

# Basic threshold
function discretizer(stream::Task, threshold::Float64)
    for r in stream
        vals = Value[]
        
        for v in r.values
            if v[2] > threshold
                push!(vals, (v[1],1.0))
            end
        end
        
        produce( Row(r.labels, r.qid, vals) ) 
    end
end

# Equal width bins
function discretizer(stream::Task, low::Float64, high::Float64, bins::Int64)

    mult = (bins-1)/(high - low)

    for r in stream
        vals = Value[]
        
        for v in r.values
            if v[2] > low
                push!(vals, (v[1], 1+floor(clamp((v[2] - low) * mult, 0, bins)) + 1) )
            end
        end
        
        produce( Row(r.labels, r.qid, vals) ) 
    end

end

# Threshold with cdf function
function discretizer(stream::Task, dist, threshold::Float64, bins::Int64)
    
    mult = bins/(1 - threshold)
    
    for r in stream
        vals = Value[]
        
        for v in r.values
            vprob = cdf(dist, v[2])
            if vprob > threshold
                push!(vals, (v[1], 1+floor((vprob-threshold)* mult)))
            end
        end
        
        produce( Row(r.labels, r.qid, vals) ) 
    end
end

function discretizer(args::Dict)

    if args["estdata"] == nothing || !isreadable(args["estdata"])
    
        if args["threshold"] == nothing
            return (s) -> s
        else
            return (s) -> @task discretizer(s, args["threshold"])
        end
        
    else
        
        if args["cdfpath"] != nothing && isreadable(args["cdfpath"])
            
            f = open(args["cdfpath"], "r")
            dfit = deserialize(f)
            close(f)
            return (s) -> @task discretizer(s, dfit, args["threshold"], args["bins"])
        
        else
            write(STDERR, "Reading data for discretization estimate ... ")
            allVals = Float64[]
            
            estf = open(args["estdata"], "r")
            for row in SparseData.readsparse(estf)
                for v in row.values
                    push!(allVals, v[2])
                end
            end
            close(estf)
            write(STDERR, "done!\n")
            
            if args["cdf"] == nothing

                sort!(allVals)
                return (s) -> @task discretizer(s, max(allVals[1], args["threshold"]), allVals[end], args["bins"])
            
            else
                write(STDERR, string("\tfitting to ",args["cdf"], " distribution\n"))
                d = eval(symbol(args["cdf"]))
                dfit = fit_mle(d, allVals)
                if args["threshold"] == nothing
                    args["threshold"] = 0.5
                end

                if args["cdfpath"] != nothing
                    f = open(args["cdfpath"], "w")
                    serialize(f, dfit)
                    close(f)
                end
                
                return (s) -> @task discretizer(s, dfit, args["threshold"], args["bins"])
            end
                
        end
    end
end

export discretizer, readsparse, Row, Value, DataStats, getStats
end # module
