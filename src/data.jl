module Data

typealias Value (Int64,Float64)

type Row
    labels::IntSet
    qid::Int64
    num::Int64
    values::Vector{Value}

    nbins::Int64
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

type Shape
    classes::Int64
    features::Int64
    unique::Int64
end

type Dataset
    shape::Shape
    ext::String
    length::Int64
    classname::Vector{String}
    sources::Vector{String}

    function Dataset(sources::Vector{String})
        ds = Dataset(sources[1])
        ds.sources = sources
        return ds
    end

    function Dataset(sources::Vector{String}, classname::String)
        ds = Dataset(sources[1], classname)
        ds.sources = sources
        return ds
    end

    function Dataset(source::String)

        ds = new(Shape(0,0,1), getExt(source), 0, [], [source])

        open(source) do f
            if ds.ext == "dat"
                l = readline(f)
                if beginswith(l, "#")
                    # expecting "# Shape 1,2,3"
                    sizes = map(parseint, split(split(strip(l)," ")[end],","))
                    if length(sizes) == 3
                        ds.shape = Shape(sizes[1], sizes[2], sizes[3])
                    else
                        ds.shape = Shape(sizes[1], sizes[2], 1)
                    end
                else
                    seekstart(f)
                    stats = getStats(f) # This reads the entire dataset
                    ds.length = stats.nRows
                    ds.shape = Shape(stats.nClasses, stats.nFeatures, stats.maxVal)
                end

                ds.classname = [ "$i" for i=1:ds.shape.classes ]

            elseif ds.ext == "arff"
                nattrs = 0
                maxn = 0
                prevmax = 0
                nclasses = 0

                for l in eachline(f)
                    l=lowercase(l)
                    if beginswith(l, "@attribute")

                        ds.classname = [ strip(v) for v in split(l[ searchindex(l,"{")+1 : rsearchindex(l,"}")-1 ], ",") ]

                        nattrs += 1
                        nclasses = length(ds.classname)
                        maxn = max(maxn, prevmax)
                        prevmax = nclasses
                    elseif beginswith(l, "@data")
                        break
                    end
                end
                ds.shape = Shape(nclasses, nattrs, maxn)
            end
        end

        return ds
    end

    function Dataset(source::String, classname::String)
        ds = Dataset(source)

        if length(classname) > 0
            names = ["" for i=1:ds.shape.classes]
            open(classname) do f
                for l in eachline(f)
                    n,name = split(strip(l), ",")
                    names[ parseint(n) ] = name
                end
            end
            ds.classname = names
        end

        return ds
    end
end

function rowAsVec(r::Row, v::Vector{Float64})
    fill!(v, 0.0)
    for p in r.values
        v[ p[1] ] = p[2]
    end
    return v
end

function rowAsVec(r::Row)
    v = spzeros(1, r.values[end][1])
    for p in r.values
        v[ p[1] ] = p[2]
    end
    return v
end

function rowAsVec(r::Row, len::Int64)
    v = spzeros(1, len)
    for p in r.values
        v[ p[1] ] = p[2]
    end
    return v
end

function vecAsRow(vec::Vector{Float64})

  r = Row(IntSet(),0,0,[],1)
  for i=1:length(vec)
    if vec[i] != 0
      push!(r.values, (i,vec[i],) )
    end
  end
  return r
end

function getExt(name::String)
    return name[ rsearchindex(name,".")+1 : end ]
end

function arffParse(line::ASCIIString, classMap::Dict{ASCIIString,Int64}, rowNum::Int64)

    row = Row(IntSet(), 0, rowNum, [], 1)

    if beginswith(line,"{")


        kstart = 1
        knext = 0
        while kstart > 0
            knext = searchindex(line,",", kstart+1)
            kmid = searchindex(line," ", kstart)

            kID = parseint(line[kstart+1 : kmid-1])

            if knext > 0
                kVal  = line[kmid+1 : knext-1]
                push!(row.values, (kID+1, parsefloat(kVal)))
            else
                # the end-2 part skips the new line and the final "}"
                push!(row.labels, get!(classMap, line[kmid+1:end-2], length(classMap)+1))
            end
            kstart = knext
        end

    else

        i=1
        k=1

        while k > 0
            kNext = searchindex(line, ",", k+1)
            if kNext == 0
                push!(row.labels, get!(classMap, line[k+1:end-1], length(classMap)+1))
            else
                iVal = parsefloat(line[k+1:kNext-1])
                if iVal != 0.0
                  push!(row.values, (i,iVal) )
                end
            end
            i += 1
            k = kNext
        end

    end

    return row
end

function svmlightParse(line::ASCIIString, nClasses::Int64, n::Int64)
    row = Row(IntSet(),0,n,[],1)
    featStart = rsearchindex(line, " ", searchindex(line, ":"))
    # Get labels
    for l in split(strip(line[1:featStart-1]), " ")
        lInt = parseint(l)
        if lInt != 0
            lInt = lInt < 0 ? nClasses + lInt + 1 : lInt
            push!(row.labels, lInt)
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
        if featEnd > 0
            push!(row.values, ( parseint(line[featStart:featMid-1]), parsefloat(line[featMid+1:featEnd]) ) )
        else
            push!(row.values, ( parseint(line[featStart:featMid-1]), parsefloat(line[featMid+1:end]) ))
        end
        featStart = featEnd
    end

    return row
end

function svmlightNext(stream::IO, nClasses::Int64, n::Int64)
    line = readline(stream)
    # Skip comments
    while !eof(stream) && ((length(line) == 0) || (line[1] == '#'))
        line = readline(stream)
    end

    if length(line)==0
        return nothing
    else
        return svmlightParse(line, nClasses, n)
    end
end

function arffNext(stream::IO, classMap::Dict{ASCIIString,Int64}, rowNum::Int64)
    line = readline(stream)
    # Skip comments
    while !eof(stream) && ((length(line) == 0) || (line[1] == '%'))
        line = readline(stream)
    end

    if length(line)==0
        return nothing
    else
        return arffParse(line, classMap, rowNum)
    end
end

function svmlightIter(stream::IO, nClasses::Int64, nbins::Int64, closeOnEnd::Bool)
    rownumber=1
    ptime = time()
    row = svmlightNext(stream, nClasses, rownumber)
    while row != nothing
        row.nbins = nbins
        produce( row )
        try
            rownumber +=1
            row = svmlightNext(stream, nClasses, rownumber)
        catch y
            println(STDERR, string(rownumber,": exception", y))
        end
    end

    if closeOnEnd
        close(stream)
    end
end

# ARFF has not been tested and isn't really supported yet...
function arffIter(stream::IO, nbins::Int64, closeOnEnd::Bool)

    # Move past the arff header
    lastNum = 0
    for l in eachline(stream)
        if beginswith(l, "@data") || beginswith(l, "@DATA")
            break
        elseif beginswith(l, "@attribute")
            lastNum += 1
        end
    end

    # TODO: Read the class names...
    classMap = Dict{ASCIIString,Int64}()

    rowNumber=1
    row = arffNext(stream, classMap, rowNumber)
    while row != nothing
        row.nbins = nbins
        rowNumber += 1
        produce( row )
        row = arffNext(stream, classMap, rowNumber)
    end

    if closeOnEnd
        close(stream)
    end
end

function eachrowTask(data::Dataset)
    for fname in data.sources
        open(fname) do f
            if endswith(fname,".dat")
                svmlightIter(f, data.shape.classes, data.shape.unique, false)
            elseif endswith(fname,".arff")
                arffIter(f, data.shape.unique, false)
            end
        end
    end
end

function eachrow(data::Dataset)
    return @task eachrowTask(data)
end

function eachrow(data::Dataset, verbose::Bool)
    return @task eachrowTaskVerbose(data)
end

function eachrowTaskVerbose(data::Dataset)

  start = time()
  prev = start
  i = 1000
  n = 0
  for r in eachrow(data)
    i -= 1
    if i==0 || (time()-prev) > 10
      i=1000
      lps = r.num / (time() - start)
      inst = r.num / (prev - start)
      prev =  time()
      println(STDERR,"Reading at: $(inst), or $(lps) avg. lps")
    end

    produce(r)
  end

end

function write(data::Dataset, rows::Task, toPath::String)
    write(data.shape, rows, toPath)
end

function write(shape::Shape, rows, topath::String)
    open(topath, "w") do f
        print(f, "# Shape $(s.classes),$(s.features),$(s.unique)\n")
        for row in rows
            print(f, join(row.labels, " "))
            if row.qid != 0
                q = row.qid
                print(f, " ",qid,":",q)
            end

            for (i,v) in row.values
                vInt=convert(Int64, v)
                print(f, " ",i,":",vInt)
            end
            print(f, "\n")
        end
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

function getStats(stream::IO)
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

export eachrow, write, Row, Value, DataStats, Dataset, getStats, rowAsVec, vecAsRow
end # module
