module Data

type RealValue
  feature::Int64
  index::Int64
  value::Float64
end

type DiscValue
  feature::Int64
  index::Int64
  value::Int64
end

type DisceteValueOutOfRange <: Exception end

Value = Union(RealValue,DiscValue)

type Row
  labels::IntSet
  qid::Int64
  num::Int64
  values::Vector{Value}
end

type DataStats
  nlabels::Int64
  nFeatures::Int64
  maxValue::Float64
  minValue::Float64
  density::Float64
  nRows::Int64
  nUnique::Int64
  isDisc::Bool
end

type Shape
  discFeatures::Int64 # Number of discrete features
  discWidth::Int64 # Total width of all discrete features
  features::Int64 # Total number of features
  index::Vector{Int64} # Mapping from global id to local id
  isIntAttr::Vector{Bool} # True if the discrete feature is an integer
  isReal::Vector{Bool} # Applies to all attributes
  labelAttr::Int64 # Used with data formats where labels can be any column
  labelLevels::Dict{String,Int64} # If the label is not an integer, this maps levels to ints
  labels::Int64 # Number of labels
  labelStrings::Vector{String} # String labels
  levels::Vector{Dict{String,Int64}} # Levels of a discrete attribute
  names::Vector{String} # Names of the attributes (applies to all)
  offsets::Vector{Int64} # Only applies to discrete features
  realFeatures::Int64 # Number of real features
  widths::Vector{Int64} # Only applies to discrete features

  Shape() = new(0,
                0,
                0,
                Int64[],
                Bool[],
                Bool[],
                0,
                Dict{String,Int64}(),
                0,
                String[],
                Dict{String,Int64}[],
                String[],
                Int64[],
                0,
                Int64[])
end

type Dataset
  length::Int64
  logMap::Bool
  verbose::Bool
  shape::Shape
  sources::Vector{String}

  Dataset() = new(0,
                  false,
                  false,
                  Shape(),
                  String[] )

  function Dataset(sources::Vector{String}, logMap::Bool)
    Dataset(sources, logMap, false)
  end

  function Dataset(sources::Vector{String}, logMap::Bool, verbose::Bool)
    ds = Dataset(sources[1], logMap, verbose)
    ds.sources = sources
    return ds
  end

  function Dataset(source::String, logMap::Bool, verbose::Bool)

    ds = Dataset()

    open(source) do f
      if getExt(source) == "dat"
        ds.shape = datShape(ds, f)
      elseif getExt(source) == "arff"
        ds.shape = arffShape(ds, f)
      else
        println("Sorry, ",getExt(source), " is not a known extension. Please use .dat or .arff")
      end
    end

    ds.logMap = logMap
    ds.verbose = verbose
    return ds
  end

  function Dataset(source::String, labelAttr::String, logMap::Bool)
    ds = Dataset(source, logMap)
    setLabelAttr(ds, labelAttr)
    return ds
  end

end

function datShape(ds::Dataset, f::IO)
  shape = Shape()
  l = readline(f)
  if beginswith(l, "#")
    # expecting "# Shape {1,2} (3,3,5,6)"
    sizes = map(parseint, split(l[ searchindex(a,"{")+1 : rsearchindex(a,"}")-1 ], ","))
    shape.offsets = Int64[1]
    shape.labels = sizes[1]
    shape.labelStrings = [ "$(i)" for i=1:shape.labels ]

    if searchindex(a,"(") == 0
      resize!(shape.isReal, sizes[2])
      fill!(shape.isReal, true)
      shape.discFeatures = 0
      shape.realFeatures = sizes[2]
    else
      widths = map(parseint, split(l[ searchindex(a,"(")+1 : rsearchindex(a,")")-1 ], ","))

      i = 0
      discI = 0
      realI = 0

      for w in widths
        i += 1
        if w == 0
          # Is real
          realI += 1
        else
          # Is disc
          discI += 1
          push!(shape.isIntAttr, true)
          push!(shape.widths, w)
          push!(shape.offsets, shape.offsets[end]+w)
        end

        push!(shape.index, (w == 0) ? realI : discI)
        push!(shape.isReal, w == 0)
      end

      shape.realFeatures = realI
      shape.discFeatures = discI
      shape.discWidth = shape.offsets[end]-1
    end
    shape.features = shape.realFeatures + shape.discFeatures
  else
    println("Sorry, .dat file is missing needed header of the form:\n\t# Shape {1,2} (3,3,5,6)")
    exit(1)
  end
end

function arffShape(ds::Dataset, f::IO)
  shape = Shape()
  shape.offsets = Int64[1]

  nlabels=0
  maxn = 0
  index=1

  i = 0
  discI = 0
  realI = 0

  for l in eachline(f)
    lp = split(strip(l), " ")

    if lowercase(lp[1]) == "@attribute"
      i += 1
      push!(shape.names, lp[2])
      discVals = Dict{String,Int64}()

      iIsReal = lowercase(lp[3]) == "real" || lowercase(lp[3]) == "numeric"

      if iIsReal
        realI += 1
      elseif beginswith(lp[3], "{")
        discI += 1

        for v in split(l[ searchindex(l,"{")+1 : rsearchindex(l,"}")-1 ], ",")
          discVals[ strip(v) ] = length(discVals)
        end

        w = length(discVals)

        isIntAttr = true
        try
          for T in discVals
            assert(0 <= parseint(T[1]) <= w)
          end
        catch
          isIntAttr = false
        end

        push!(shape.isIntAttr, isIntAttr)
        push!(shape.levels, discVals)
        push!(shape.offsets, shape.offsets[end]+w)
        push!(shape.widths, w)

      end

      push!(shape.isReal, iIsReal)
      push!(shape.index, iIsReal ? realI : discI)

      if lowercase(lp[2]) == "class"
        shape.labels = length(discVals)
        shape.labelAttr = i
        shape.labelStrings = [ c[1] for c in discVals ]
        sort!( shape.labelStrings, by=(x)->discVals[ x ] )
      end

    elseif lowercase(lp[1]) == "@data"
      break
    end
  end

  shape.discFeatures = discI
  shape.realFeatures = realI
  shape.features = shape.realFeatures + shape.discFeatures
  shape.discWidth = (discI > 0) ? shape.offsets[end]-1 : 0
  return shape
end

function ==(a::Shape, b::Shape)
  a.labels == b.labels &&
    a.features == a.features &&
    a.unique == a.unique
end

function setLabelAttr(ds::Dataset, labelAttr::String)
  setLabelAttr(ds, indmax(.==(ds.shape.names,labelAttr)))
end

function setLabelAttr(ds::Dataset, labelAttr::Int64)

  ds.shape.labelAttr = labelAttr
  ds.shape.labelIndex = ds.shape.levels[ ds.shape.labelAttr ]
  ds.shape.labels = [ k for k in keys(ds.shape.labelLevels) ]
  sort!(ds.shape.labels, by = (v) -> ds.shape.labelLevels[v])

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
  r = Row(IntSet(),0,0,Value[])
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

function parseValue(shape::Shape, index::Int64, val::String, logMap::Bool)
  typeid = shape.index[index]

  if shape.isReal[index]
    kFloat = parsefloat(val)
    return RealValue(index, typeid, logMap ? log(kFloat) : kFloat)
  elseif shape.isIntAttr[typeid]
    return DiscValue(index, typeid, shape.offsets[typeid]+parseint(val))
  else
    levels = shape.levels[typeid]
    if haskey(levels, val)
      return DiscValue(index, typeid, shape.offsets[typeid] + levels[val])
    else
      throw(DisceteValueOutOfRange())
    end
  end

end

function parseSVMLightRow(data::Dataset, line::ASCIIString, rowNum::Int64)
  row = Row(IntSet(), 0, rowNum, Data.Value[])

  featStart = rsearchindex(line, " ", searchindex(line, ":"))

  # Get labels
  for l in split(strip(line[1:featStart-1]), " ")
    lInt = parseint(l)
    if lInt > 0 # <=0 means no label
      assert(lInt > 0) # < 0 is not going to work either
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

    id = parseint(line[featStart:featMid-1])
    if featEnd > 0
      val = line[featMid+1:featEnd]
    else
      val = line[featMid+1:end]
    end
    push!(row.values, parseValue(data.shape, id, val, data.logMap))
    featStart = featEnd
  end

  return row
end

function svmlightNext(data::Dataset, stream::IO, rowNum::Int64)
  line = readline(stream)
  # Skip comments
  while !eof(stream) && ((length(line) == 0) || (line[1] == '#'))
    line = readline(stream)
  end
  parseSVMLightRow(data, line, rowNum)
end

function svmlightIter(data::Dataset, stream::IO, closeOnEnd::Bool)
  rowNum=0
  while !eof(stream)
    try
      rowNum +=1
      produce( svmlightNext(data, stream, rowNum) )
    catch y
      println(STDERR, string(rowNum,": exception", y))
    end
  end

  if closeOnEnd
    close(stream)
  end
end

function parseArffRow(data::Dataset, line::ASCIIString, rowNum::Int64)

  row = Row(IntSet(), 0, rowNum, Data.Value[])
  i=0

  if line[1] == '{'

    kstart = 1
    knext = 0
    while kstart > 0
      i += 1
      knext = searchindex(line,",", kstart+1)
      kmid = searchindex(line," ", kstart)
      kID = parseint(line[kstart+1 : kmid-1]) + 1
      if knext > 0
        kVal  = line[kmid+1 : knext-1]
      else
        kVal = line[kmid+1:end-2]
      end
      kstart = knext
      if kID == data.shape.labelAttr

        dID = data.shape.index[kID]
        if data.shape.isIntAttr[ dID ]
          push!(row.labels, parseint(kVal)+1)
        else
          push!(row.labels, data.shape.levels[dID][kVal]+1)
        end

      else
        push!(row.values, parseValue(data.shape, kID, kVal, data.logMap))
      end

    end

  else

    i=1
    k=1

    while k > 0
      kNext = searchindex(line, ",", k+1)
      kVal = (kNext == 0) ? line[k+1:end-1] : line[k+1:kNext-1]
      typeid = data.shape.index[i]
      if i == data.shape.labelAttr
        if data.shape.isIntAttr[typeid]
          push!(row.labels, parseint(kVal))
        else
          push!(row.labels, data.shape.levels[typeid][kVal])
        end
      else
        push!(row.values, parseValue(data.shape, i, kVal, logMap))
      end

      i += 1
      k = kNext
    end

  end

  return row
end

function arffNext(data::Dataset, stream::IO, rowNum::Int64, logMap::Bool)
  line = readline(stream)
  # Skip comments
  while !eof(stream) && ((length(line) == 0) || (line[1] == '%'))
    line = readline(stream)
  end
  parseArffRow(data, line, rowNum)
end

function arffIter(data::Dataset, stream::IO, closeOnEnd::Bool)
  rowNum=0

  line = ""
  while !beginswith(line, "@data")
    line = readline(stream)
  end

  while !eof(stream)
    rowNum +=1
    produce( arffNext(data, stream, rowNum, data.logMap) )
  end

  if closeOnEnd
    close(stream)
  end
end

function eachrowTask(data::Dataset)
  for fname in data.sources
    open(fname) do f

      if endswith(fname,".dat")
        svmlightIter(data, f, false)
      elseif endswith(fname,".arff")
        arffIter(data, f, false)
      end

    end
  end
end

function eachrow(data::Dataset)

  dTask = @task eachrowTask(data)

  if data.verbose
    return @task verbosify( dTask, "Reading" )
  else
    return dTask
  end

end

function churn(t::Task, width::Int64)
  @task churnTask(t, width)
end

function churnTask(t::Task, width::Int64)

  buf = Any[]
  for i=1:width
    push!(buf, consume(t))
  end

  for k in t
    idx = convert(Int64, ceil(rand() * width))
    produce(buf[idx])
    buf[idx] = k
  end

  for i=1:width
    produce(buf[i])
  end

end

function verbosify(t::Task, msg::String)
  start = time()
  prev = start
  i = 1000
  n = 0
  for r in t

    i -= 1
    if i==0 || (time()-prev) > 10
      i=1000
      lps = r.num / (time() - start)
      inst = r.num / (prev - start)
      prev =  time()
      println(STDERR, msg, " at: ", inst, ", or ", lps," avg. lps")
    end

    produce(r)
  end

end

function write(data::Dataset, rows::Task, toPath::String)
  write(data.shape, rows, toPath)
end

function write(shape::Shape, rows, topath::String)
  open(topath, "w") do f
    print(f, "# Shape $(s.labels),$(s.features),$(s.unique)\n")
    for row in rows
      print(f, join(row.labels, " "))
      if row.qid != 0
        q = row.qid
        print(f, " ", qid,":",q)
      end

      for v in row.values
        print(f, " ", v.index, ":", v.value)
      end
      print(f, "\n")
    end
  end
end

# Don't use this...
function getStats(stream::Task)
  labels = IntSet()
  minVal = 0 # <- 0 is the missing value, so just assume it was in there...
  maxVal = -Inf
  nfeatures = 0
  nvals = 0
  nunique = 0
  nrows = 0
  isDisc = true

  try
    for row in stream
      nrows += 1
      union!(labels, row.labels)
      for v in row.values
        nfeatures = max(nfeatures, v.index)
        nunique = max(nunique, convert(Int64, v.value))
        minVal = min(minVal, v.value)
        maxVal = max(maxVal, v.value)
        nvals += 1
      end
    end
  catch
    isDisc = false
    nunique = 0
    for row in stream
      nrows += 1
      union!(labels, row.labels)
      for v in row.values
        nfeatures = max(nfeatures, v.index)
        minVal = min(minVal, v.value)
        maxVal = max(maxVal, v.value)
        nvals += 1
      end
    end
  end

  DataStats(length(labels),
            nfeatures,
            maxVal,
            minVal,
            nvals/(nfeatures * nrows),
            nrows,
            nunique,
            isDisc)
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
