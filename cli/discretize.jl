#!/usr/bin/env julia
using Distributions
using SparseML: Data
import Base.write

istart=3
dist = nothing
destDir = "./"
mode = ""

function mklbl(t)
    return parseint(t[1]),t[2]
end

if length(ARGS) == 0
    println("Usage: discretize.jl [output dir] [data.features] training.dat [ more data files, which will not be used to find bins... ]")
    quit()
end

fileStart=1

if isdir(ARGS[fileStart])
    destDir = ARGS[fileStart]
    fileStart += 1
else
    destDir = dirname(ARGS[fileStart])
end

if endswith(ARGS[fileStart],".features")

    # Write feature names if we have a feature names file
    if endswith(ARGS[fileStart],".features")
        open(ARGS[fileStart]) do f
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

    fileStart += 1
end

files = [ ARGS[j] for j=fileStart:length(ARGS) ]

baseDat = Data.Dataset(files[1])

vsets  = [ Set{Float64}(0.0) for i=1:baseDat.shape.features ]
dmins  = [ 0.0 for i=1:baseDat.shape.features ]
dmaxes = [ 0.0 for i=1:baseDat.shape.features ]

line_num = 0
notify_counter = 1000
start_time = time()

for r in Data.eachrow(baseDat)

    line_num += 1

    for (i,v) in r.values
        dmins[i]  = min(dmins[i],  v)
        dmaxes[i] = max(dmaxes[i], v)
        push!(vsets[i], v)
    end

    notify_counter -= 1
    if notify_counter == 0
      tdelta = time() - start_time
      lps = line_num / tdelta
      notify_counter = 1000
      println("$(line_num): Reading at an average of $(lps) lps")
    end

end

delta = time() - start_time
println("Done finding limits in $(delta) seconds, determining bins...")

nbins = [ convert(Int64, max(1, round(2*log(length(s))))) for s in vsets ]
maxBins = maximum(nbins)
vsets = nothing # Free memory...

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

    fbase = files[i][ 1 : rsearchindex(files[i], ".") - 1 ]

    denseName = "$(fbase)_ordinal.dat"
    binaryName = "$(fbase)_binary.dat"

    line_num = 0
    notify_counter = 1000

    open( joinpath(destDir, denseName), "w") do denseOut
        open( joinpath(destDir, binaryName), "w") do binaryOut

            write(denseOut, string("# Shape $(baseDat.shape.classes),$(ndenseFeatures),$(maxBins) \n"))
            write(binaryOut, string("# Shape $(baseDat.shape.classes),$(nbinaryFeatures),1 \n"))

            start_time = time()

            for r in Data.eachrow(datainput)
                line_num += 1

                rlabels = join(r.labels, " ")
                write(denseOut, rlabels)
                write(binaryOut, rlabels)

                for (vi,val) in r.values
                    val = clamp(val, dmins[vi], dmaxes[vi])
                    discval = convert(Int64, round((val-dmins[vi]) * dranges[vi] ))
                    binaryVal = convert(Int64, startIndex[vi] + discval)

                    if discval > 0
                        write(denseOut, " $(vi):$(discval)")
                        write(binaryOut, " $(binaryVal):1")
                    end

                end

                write(denseOut, "\n")
                write(binaryOut, "\n")

                notify_counter -= 1
                if notify_counter == 0
                  tdelta = time() - start_time
                  lps = line_num / tdelta
                  notify_counter = 1000
                  println("Discretizing at $(lps) lps")
                end

            end

        end
    end

end
