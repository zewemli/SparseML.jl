#!/usr/bin/env julia
using Distributions
using SparseML: Data

istart=3
dist = nothing
destDir = "./"
mode = ""

function mklbl(t)
    return parseint(t[1]),t[2]
end

if length(ARGS) == 0
    println("Usage: svmDiscretize.jl [output dir] [data.features] training.dat [ more data files, which will not be used to find bins... ]")
    quit()
end

fileStart=1

if isdir(ARGS[fileStart])
    destDir = ARGS[fileStart]
    fileStart += 1
else
    destDir = dirname(ARGS[fileStart])
end

if !endswith(ARGS[fileStart],".dat")
    fileStart += 1
end

files = [ ARGS[j] for j=fileStart:length(ARGS) ]

baseDat = Data.Dataset(files[1])

vsets  = [ Set{Float64}(0.0) for i=1:baseDat.shape.features ]
dmins  = [ 0.0 for i=1:baseDat.shape.features ]
dmaxes = [ 0.0 for i=1:baseDat.shape.features ]

for r in Data.iter(baseDat)
    for (i,v) in r.values
        dmins[i]  = min(dmins[i],  v)
        dmaxes[i] = max(dmaxes[i], v)
        push!(vsets[i], v)
    end
end

close(baseDat.handle)

nbins = [ convert(Int64, max(1, round(2*log(length(s))))) for s in vsets ] 
maxBins = maximum(nbins)
vsets = nothing # Free memory...

# Write feature names if we have a feature names file
if !endswith(ARGS[1],".dat")
    open(ARGS[1]) do f
        i=0
        for l in eachline(f)
            id,name = split(strip(l),",")
            id = parseint(id)
            id_nbins = nbins[id]
            for n=1:id_nbins
                i += 1
                write(STDOUT, "$i,$(name):b$(n)_of_$(id_nbins)\n")
            end
        end
    end
end

dranges = ./(nbins, .-(dmaxes, dmins))
nbinaryFeatures = sum(nbins)
ndenseFeatures  = length(nbins)

startIndex = zeros(length(nbins))
idx=1
for i=1:length(nbins)
    startIndex[i] = idx
    idx += nbins[i]
end

for i=1:length(files)

    datainput = Data.Dataset(files[i])
    
    fbase = replace(basename(files[i]), ".dat","")
    
    denseName = "$(fbase)_dense.dat")
    binaryName = "$(fbase)_binary.dat"
    
    open( joinpath(destDir, denseName), "w") do denseOut
        open( joinpath(destDir, binaryName), "w") do binarydat
    
            write(denseOut, string("# Shape $(baseDat.shape.classes),$(ndenseFeatures),$(maxBins) \n"))
            write(binarydat, string("# Shape $(baseDat.shape.classes),$(nbinaryFeatures),1 \n"))
            
            for r in Data.iter(datainput)
                rlabels = join(r.labels, " ")
                write(outdat, rlabels)
                write(denseOut, rlabels)
                
                for (vi,val) in r.values
                    val = clamp(val, dmins[vi], dmaxes[vi])
                    discval = convert(Int64, round((val-dmins[vi]) * dranges[vi] ))
                    
                    binaryVal = startIndex[vi] + discval
                    
                    if discval > 0
                        write(denseOut, " $vi:$discval")
                    end
                    write(binarydat, " $binaryVal:1")
                    
                end
                
                write(denseOut, "\n")
                write(binarydat, "\n")
                
            end
            
        end
    end
    
    close(datainput.handle)
end
