println("Loading Python stuff")

using PyCall
using PyPlot
@pyimport numpy as np
@pyimport cv2
using Images
using ImageMagick

function get_default_options()
    opts = Dict("dimX"=>224, "dimY"=>224, "numChannels"=>4, "numSlices"=>161, "channel"=>4, "templateFrame"=>1, "timePts"=>500, "templateTime"=>1, "type"=>Array{UInt16, 5}, "startFrame"=>1, "debug"=>true)
    get!(opts, "stopFrame", opts["timePts"])
    get!(opts, "format", (opts["dimX"], opts["dimY"], opts["numChannels"], opts["numSlices"], opts["timePts"]))
    opts
end

function fix(x)
    if x>0 && x!=Int
        x = floor(x)
    elseif x<0 && x!=Int
        x = ceil(x)
    else
        x
    end
    Int(x)
end

function adjust_t(t0, dim)
    if t0 > fix(dim/2)
        t = t0 - dim- 1
    else
        t = t0 - 1
    end
    t
end

function phase_corr_reg(F0, F)
    X = F0.*conj(F)
    X = ifft(X/norm(X))
    max, argmax = findmax(abs.(X))
    tx, ty = map(x->ind2sub(X, x), argmax)
    m , n = size(F0)
    tx = adjust_t(tx, m)
    ty = adjust_t(ty, n)
    (tx, ty, m, n)
end

function regseq2(inpath, outpath, opts = [])

    if length(opts) == 0
        opts = get_default_options()
    end

#    if opts["debug"]
#        tic()
#    end

    println("Loading image")

    infile = load(inpath)
    data = Array{Float64, 3}(infile)

    println("Computing template")
    
    img1 = data[:, :, opts["templateFrame"]]
    templatefft = fft(img1)
    ts = Any[]
    outfile = open(outpath, "w")
    for frame = opts["startFrame"]:opts["stopFrame"]
       	print("frame = ", frame)
        img2 = data[:, :, frame]
	tx, ty, m, n = phase_corr_reg(templatefft, fft(img2))
	println(", tx = ", tx, ", ty = ", ty)
	push!(ts, [tx,ty])
	M = np.float64([[1, 0, tx]'; [0, 1, ty]'])
	newimg = cv2.warpAffine(img2, M, (m, n))
#	imshow(newimg)
#	show()
        write(outfile, newimg)
	if opts["debug"]
	    # fprintf... finish this part later
	end
    end
end

opts = Dict("dimX"=>512, "dimY"=>512, "timePts"=>500, "templateFrame"=>25, "startFrame"=>1, "debug"=>true)
get!(opts, "stopFrame", opts["timePts"])
regseq2("smaller.tif", "newsmaller.raw", opts)



# Problem on line 65 either with the Float64 or the "vec" part