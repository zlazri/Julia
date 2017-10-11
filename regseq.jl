using PyCall
using PyPlot
@pyimport numpy as np
@pyimport cv2

function get_default_options()
    opts = Dict("dimX"=>224, "dimY"=>224, "numChannels"=>4, "numSlices"=>161, "channel"=>4, "templateFrame"=>25, "timePts"=>50, "templateTime"=>1, "type"=>Array{UInt16, 5}, "startFrame"=>1, "debug"=>true)
    get!(opts, "stopFrame", opts["timePts"])
    get!(opts, "format", (opts["dimX"], opts["dimY"], opts["numChannels"], opts["numSlices"], opts["timePts"]))
    opts
end

function preproc(inp)
    vec = inp[:]
    out = float(inp - minimum(vec))/float(maximum(vec)-minimum(vec))
end

function postproc(inp)
    out = inp/maximum(inp[:])
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
    X = ifft(F0.*conj(F), 2)
    max1, argmax1 = findmax(abs(X), 1)
    remap = map(x->ind2sub(X, x), argmax1)
    max2, argmax2 = findmax(max1)
    max1 = X[argmax1]
    max2 = max1[argmax2]
    tx = argmax2
    ty = remap[argmax2][1]
    m , n = size(F0)
    tx = adjust_t(tx, m)
    ty = adjust_t(ty, n)
    (tx, ty, m, n)
end

function regseq(inpath, outpath, options = [])

    if length(options) == 0
        opts = get_default_options()
    end

#    if opts["debug"]
#        tic()
#    end

    f = open(inpath)
    infile = Mmap.mmap(f, opts["type"], opts["format"])
    img1 = preproc(infile[:, :, opts["channel"], opts["templateFrame"], opts["templateTime"]])
    templatefft = fft(img1, 2)

    outfile = open(outpath, "w")
    for frame = opts["startFrame"]:opts["stopFrame"]
        img2 = preproc(infile[:, :, opts["channel"], frame, opts["templateTime"]])
	tx, ty, m, n = phase_corr_reg(templatefft, fft(img2, 2))
	M = np.float64([[1, 0, tx]'; [0, 1, ty]'])
	newimg = cv2.warpAffine(img2, M, (m, n))
	figure(frame)
	write(outfile, postproc(newimg))
#	if opts["debug"]
	    # fprintf... finish this part later
#	end
    end
end

regseq("/home/zlazri/Documents/Vol_Data/Image_0001_0001.raw", "ModifiedTempImgs.raw")



# Problem on line 65 either with the Float64 or the "vec" part