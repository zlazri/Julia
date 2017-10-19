# GAUSSIAN FILTER FOR 2D LAPLACIAN PYRAMID. BLOB DETECTOR

println("Loading libraries")

using Combinatorics
using PyPlot
using PyCall
@pyimport matplotlib.patches as patch
circle = patch.pymember("Circle")
#using ImageView

println("Done Loading libraries")

f = open("/home/zlazri/Documents/2Ddata/averageimg.raw")

width = 512
height = 512

img = Mmap.mmap(f, Array{Float64, 2}, (width, height))

function getbinomkern(ord)
    assert(ord > 0)
    b1 = Array{Float64}([0.25; 0.5; 0.25])
    k = copy(b1)
    curord = 1
    while curord < ord
        k = conv(b1, k)
        curord += 1
    end
    k
end

function binomfilt2{T}(x::Array{T, 2}, ord::Int)
    k = getbinomkern(ord)

    M, N = size(x)

    X = fft(x, 1)

    kpad1 = zeros(Complex128, M)
    kpad2 = zeros(Complex128, N)

    kpad1[1:Int(ceil(length(k)/2))] = k[Int(ceil(length(k)/2)):Int(length(k))]
    kpad1[Int(M-floor(length(k)/2)+1):M] = k[1:Int(floor(length(k)/2))]
    fft!(kpad1)
    for n = 1:N X[:, n].*= kpad1 end

    ifft!(X, 1)
    fft!(X, 2)

    kpad2[1:Int(ceil(length(k)/2))] = k[Int(ceil(length(k)/2)):Int(length(k))]
    kpad2[Int(N-floor(length(k)/2)+1):N] = k[1:Int(floor(length(k)/2))]
    fft!(kpad2)
    for m = 1:M X[m, :].*= kpad2 end

    real(ifft(X, 2))
end


decimg{T}(x::Array{T, 2}) = x[map(m -> 1:2:m, size(x))...]

function interpimg{T}(x::Array{T, 2}, interpfilt)
    y = zeros(T, map(x -> 2*x, size(x)))
    y[map(m -> 1:2:2m, size(x))...] = x
    interpfilt(y)
end


function interpbinom{T}(x::Array{T, 2}, ord::Int=1)
    interpimg(x, y -> binomfilt2(y, ord))
end

function argmax{T}(x::Array{T, 2})
    m1, m2 = size(x)
    maxval = -Inf
    arg = (0, 0, 0)
    for i = 1:m1, j = 1:m2
        val = x[i, j]
	if val > maxval
	    arg = (i, j)
	end
	maxval = max(val, maxval)
    end
    arg
end


function arglocalmax{T}(x::Array{T, 2}, r=1, thresh=100)
    m1, m2 = size(x)
    I = -r:r
    i0 = r+1
    center = (i0, i0)
    maxs = Dict{Tuple{Int, Int}, T}()
    for i = (r+1):(m1-r-1), j = (r+1):(m2-r-1)
        if argmax(x[i + I, j + I]) == center
	    value = x[i, j]
	    if value >= thresh
	        maxs[(i, j)] = x[i, j]
	    end
	end
    end
    maxs
end


function getlocalmaxs(pyr::Array{Any, 1}, levels, threshes)
    maxs = []
    for l = 1:levels
        for ((i, j), value) = arglocalmax(pyr[l], 2, threshes[l])
	    push!(maxs, (i, j, value, l))
	end
    end
    ordering = tup -> (tup[3], tup[4])
    sort!(maxs, by=ordering, rev=true)
    println("Local maxs found and sorted--by value and level--for all levels of Pyramid")
    maxs
end


function makelappyr{T}(img::Array{T, 2}, ord::Int=1)
    L = Int(ceil(log2(minimum(size(img)))) + 1)
    pyr = Any[img]
    for l = 2:L
        if any(m -> m <= ord, size(pyr[l - 1]))
            break
        end
        push!(pyr, decimg(binomfilt2(pyr[l - 1], ord)))
        up = interpbinom(pyr[l], ord)
        pyr[l - 1] -= up[map(m -> 1:m, size(pyr[l - 1]))...]
    end
    println("Pyramid created")
    pyr
end


function postprocpyr(pyr::Array{Any, 1})
    numlevels = length(pyr)
    for i = 1:numlevels
        img = pyr[i]
	vec = img[:]
	out = Float64.(img - minimum(vec))/Float64(maximum(vec) - minimum(vec))
	pyr[i] = out
    end
    println("Pyramid normalized")
    pyr
end

#------------------Test-------------------------------------------
pts_1 = 5

i_1 = 6

sig_1 = 5

pyr_1 = makelappyr(img)

#pyr_1 = makelappyr(img, pts_1, i_1, sig_1)

pyr_1 = postprocpyr(pyr_1)

levels_1 = size(pyr_1)[1]

threshes_1 = [0.25, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

loc_maxs_1 = getlocalmaxs(pyr_1, levels_1, threshes_1)

#-----------------------Display Image Pyramid-----------------------

function getrowpad(pyr::Array{Any, 1}, level::Int64)

    assert(level>1)

    if level == 2
        pad = 0

    else
        pad = 0

        for i = 1:level-2
	    pad = pad + size(pyr[i+1])[1]
	end
	
    end

    pad

end


function imgpyrdisplay(pyr::Array{Any, 1})

    display = zeros(size(pyr[1])[1], size(pyr[1])[2]+size(pyr[2])[2])
    for level = 1:length(pyr)

        if level == 1
	    m, n = size(pyr[level])
	    display[1:m, 1:n] = pyr[level]

        else
	    rowpad = 1+getrowpad(pyr, level)
	    colpad = 1+size(pyr[1])[2]
	    rowfill = rowpad+size(pyr[level])[1]-1
	    colfill = colpad+size(pyr[level])[2]-1
            display[rowpad:rowfill, colpad:colfill] = pyr[level][:, :]
	end
    end

    imshow(display)
    show()

end

#---------------------------------------------------------------

#imgpyrdisplay(pyr_1)


#------------Blob Detection and Trimming------------------------

function blob_upscale(loc_max::Array{Any, 1}, ord::Int=1)

#   Scales the coordinates to the base level based on their level in the pyramid. Also scales blob size.

    sig_sq = (ord * (1/4))+(1/4) 
    sig = sqrt(sig_sq)    # blob size for base of the pyramid

    scaled_pts = []
    for i = 1:length(loc_max)
        level = loc_max[i][4]
	scale = 2^(level-1)
	scaled_tup = (scale*loc_max[i][1], scale*loc_max[i][2], loc_max[i][3], loc_max[i][4], scale*sig)
	push!(scaled_pts, scaled_tup)
    end
    println("Sizes of the blobs have been scaled appropriately")
    scaled_pts
end

#--------------------------------------------------------------

loc_max_new = blob_upscale(loc_maxs_1)
#print(loc_max_new, ",")

#--------------------------------------------------------

function blob_overlap{T}(blob1::Array{T, 1}, blob2::Array{T, 1}, sig1::Float64, sig2::Float64)

#   Determines the amount of overlap between two 2D blobs.

    x1 = blob1[1]
    y1 = blob1[2]

    x2 = blob2[1]
    y2 = blob2[2]

    r1 = sig1*sqrt(2)
    r2 = sig2*sqrt(2)
    
    d = sqrt((x1-x2)^2 + (y1-y2)^2)

    if d > r1 + r2
        return 0
    end

    if d <= abs(r1 - r2)
	return 1
    end

    ratio1 = ((d^2) + (r1^2) - (r2^2))/(2 * d * r1)
    ratio2 = ((d^2) + (r2^2) - (r1^2))/(2 * d * r2)
    a = -d + r1 + r2
    b = d + r1 - r2
    c = d - r1 + r2
    d = d + r1 + r2

    intersect_area = (r1^2) * acos(ratio1) + (r2^2) * acos(ratio2) - (1/2) * sqrt(a*b*c*d)

    radii = [r1, r2]
    radius = minimum(radii)
    area = pi*(radius^2)

    return intersect_area/area
end

function blob_trimmer(x::Array{Any, 1}, threshold = 0.2)

    # Determines the amount of overlap between two blobs. Kills blobs that overlap too much.

    println("oldsize: ", size(x))
    allblobcombs = combinations(x,2)
    allblobocmbs = collect(allblobcombs)
    b = 0
    for comb in allblobcombs
    
        coord1 = [comb[1][1], comb[1][2]]
	coord2 = [comb[2][1], comb[2][1]]
	over = blob_overlap(coord1, coord2, comb[1][5], comb[2][5])
	
	if over > threshold
	            
            if comb[1][3] > comb[2][3]
	        x = filter(y -> y != comb[2], x)
            else
	        x = filter(y -> y != comb[1], x)
            end
	    println("filtering x: ", size(x))
	end
	
    end
    println("newsize: ", size(x))
    x
end

#-----------------------------------------------------------------

blobs_trimmed = blob_trimmer(loc_max_new)
print(size(loc_max_new))
#print(blobs_trimmed)

#--------------------------Plot Blobs-------------------------------
println("blobtrimmed size: ", size(blobs_trimmed))
fig, ax = subplots(1)
imshow(img)
ax = gca()
blob_coors = []
for blob in blobs_trimmed
    coorstup = (blob[1], blob[2])
    push!(blob_coors,coorstup)
    c = circle((blob[2], blob[1]), 2, edgecolor = "black", facecolor = "none")
    c[:radius] = blob[5]*sqrt(2)
    ax[:add_patch](c)
end
show()

#-----------------------Continous Scale Parameter-------------------

function upsample(pyr::Array{Any, 1}, l::Int64, numupsamples::Int64, pts, i, sig)

    # upsamples an image (level of the pyramid) a specific number of times.

    currentimg = pyr[l]
    for k = 1:numupsamples
        currentimg = interpgauss(currentimg, pts, i, sig)
    end
    currentimg
end


function get_locvals_upperlevels(pyr::Array{Any, 1}, numlevels::Int64, locmaxinfo::Tuple{Int64, Int64, Float64, Int64, Int64}, pts, i, sig)

    # Takes in a local max for a specific level and finds the corresponding values for this local max at levels above it. Then it returns the values for levels above it, along with their corresponding sigma values.

    l = locmaxinfo[4]
    numupperlevels = numlevels - 1
    upperlevelmaxs = []
    img = pyr[l]
    currentsig = locmaxinfo[5]
    coors = [locmaxinfo[1], locmaxinfo[2]]./(2^(l-1))

    for k = 1:numupperlevels
        scale = k
	upsampledimg = upsample(pyr, locmaxinfo[4] + k, k, pts, i, sig)
	vallocmax = upsampledimg[Int(coors[1]), Int(coors[2])]
	newsig = currentsig*(2^scale)
	sig_tip = (newsig, vallocmax)
	push!(upperlevelmaxs, sig_tup)
    end
    upperlevelmaxs
end


function get_locvals_lowerlevels(pyr::Array{Any, 1}, numlevel::Int64, locmaxinfo::Tuple{Int64, Int64, Float64, Int64, Int64}, pts, i, sig)

    # Takes in a local max for a specfic level and finds the corresponding values for this local max at levels below it. Then it returns the values for the levels below it, along with their corresponding sigma values.

    l = locmaxinfo[4]
    numlowerlevels = l - 1
    lowerlevelmaxs = []
    img = pyr[l]
    currentsig = locmaxinfo[5]
    coors = [locmaxinfo[1], locmaxinfo[2]]./(2^(l-1))

    for k = 1:numlowerlevels
        currentimg = pyr[k]
	scale = l-k
	newcoors = coors.*(2^scale)
	vallocmax = currentimg[Int(newcoors[1]), Int(newcoors[2])]
	newsig = currentsig/(2^scale)
	sig_tup = (newsig, vallocmax)
	push!(lowerlevelmaxs, sig_tup)
    end
    lowerlevelmaxs
end


function lagrange_interp{T}(x::Array{T, 1}, y::Array{T, 1}, u::Array{Float64, 1})

    # Performs Lagrange interpolation. x is the array of points you have, y is the array of values corresponding to the function at the x points, and u is the array of points for which you are looking for function values.

    n = length(x)
    v = zeros(size(u))
    for k = 1:n
        w = ones(size(u))
	for j = 1:n
	    if k == j
	        continue
            else
	        w = (u-x[j])./(x[k]-x[j]).*w
	    end
	end
        v = v + w*y[k]
    end
    v
end


function continuous_sig(pyr::Array{Any, 1}, loc_maxs::Array{Any, 1}, pts, i, sig)

    # Uses interpolation to create a continuous function out or the scale parameter to determine the appropriate size of a blob a blob for a given local max.

    new_blobs_trimmed = []
    numlocmaxs = length(loc_maxs)
    levels = length(pyr)
    plotticks = linspace(1, pyr_levels, 350)
    plotticks = Array(plotticks)

    for k = 1:numlocmaxs
        currentmax = loc_maxs[k]
	upperlevels = get_locvals_upperlevels(pyr, levels, currentmax, pts, i, sig)
	lowerlevels = get_locvals_lowerlevels(pyr, levels, currentmax, pts, i, sig)
	levelmaxs = []

        for j = 1:length(lowerlevels)
	    push!(levelmaxs, lowerlevels[j])
	end

        push!(levelmaxs, lowerlevels[j])

        for j = 1:length(upperlevels[j])
	    push!(levelmaxs, upperlevels[j])
	end

        maxsvec = []
	points = []
	for j = 1:levels
	    push!(maxsvec, levelmaxs[j][2])
	    push!(points, j)
	end
        f = lagrange_interp(points, maxsvec, plotticks)

        val, index = findmax(f)
	maxpoint = plotticks[index]
	push!(new_blobs_trimmed, (loc_maxs[k][1], loc_maxs[k][2], val, maxpoint, sig*2^(maxpoint-1)))
    end
    new_blobs_trimmed
end

#---------------------------------------------------------------------------

#new_trimmed_blobs = continuous_sig(pyr_1, blobs_trimmed, pts_1, i_1, sig_1)
#print(new_trimmed_blobs)
