# BINOMIAL FILTER FOR 3D LAPLACIAN PYRAMID. BLOB DETECTOR

println("Loading libraries")

using Combinatorics
#using ImageView
using PyPlot
using PyCall
@pyimport mayavi
@pyimport mayavi.mlab as mlab
@pyimport matplotlib.patches as patch
circle = patch.pymember("Circle")

println("Done loading libraries")

f = open("/home/zlazri/Documents/Vol_Data/Image_0001_0001.raw")

width = 224
height = 224
nslices = 161
nchans = 4
tsteps = 50
m = Mmap.mmap(f, Array{UInt16, 5}, (width, height, nchans, nslices, tsteps))

chan = 4

getvol(tstep::Int) = m[:, :, chan, 1:nslices, tstep]

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

function binomfilt3{T}(x::Array{T, 3}, ord::Int)

    k = getbinomkern(ord)
    
    M, N, D = size(x)

    X = fft(x, 1)
    
    kpad1 = zeros(Complex128, M)
    kpad2 = zeros(Complex128, N)
    kpad3 = zeros(Complex128, D)

    kpad1[1:Int(ceil(length(k)/2))] = k[Int(ceil(length(k)/2)):Int(length(k))]
    kpad1[Int(M-floor(length(k)/2)+1):M] = k[1:Int(floor(length(k)/2))]
    fft!(kpad1)
    for n = 1:N, d = 1:D X[:, n, d] .*= kpad1 end

    ifft!(X, 1)
    fft!(X, 2)

    kpad2[1:Int(ceil(length(k)/2))] = k[Int(ceil(length(k)/2)):Int(length(k))]
    kpad2[Int(N-floor(length(k)/2)+1):N] = k[1:Int(floor(length(k)/2))]
    fft!(kpad2)
    for m = 1:M, d = 1:D X[m, :, d] .*= kpad2 end

    ifft!(X, 2)
    fft!(X, 3)

    kpad3[1:Int(ceil(length(k)/2))] = k[Int(ceil(length(k)/2)):Int(length(k))]
    kpad3[Int(D-floor(length(k)/2)+1):D] = k[1:Int(floor(length(k)/2))]
    fft!(kpad3)
    for m = 1:M, n = 1:N X[m, n, :] .*= kpad3 end

    real(ifft(X, 3))
end

decvol{T}(x::Array{T, 3}) = x[map(m -> 1:2:m, size(x))...]

function interpvol{T}(x::Array{T, 3}, interpfilt)
    y = zeros(T, map(x -> 2*x, size(x)))
    y[map(m -> 1:2:2m, size(x))...] = x
    interpfilt(y)
end

function interpbinom{T}(x::Array{T, 3}, ord::Int)
    interpvol(x, y -> binomfilt3(y, ord))
end

function argmax{T}(x::Array{T, 3})
    m1, m2, m3 = size(x)
    maxval = -Inf
    arg = (0, 0, 0)
    for i = 1:m1, j = 1:m2, k = 1:m3
        val = x[i, j, k]
        if val > maxval
            arg = (i, j, k)
        end
        maxval = max(val, maxval)
    end
    arg
end

function arglocalmax{T}(x::Array{T, 3}, r=1, thresh = 900, normalize::Bool = true)
    m1, m2, m3 = size(x)
    I = -r:r
    i0 = r + 1
    center = (i0, i0, i0)
    maxs = Dict{Tuple{Int, Int, Int}, T}()
    maxs_array = []
    points = []
    for i = (r+1):(m1-r-1), j = (r+1):(m2-r-1), k = (r+1):(m3-r-1)
        if argmax(x[i + I, j + I, k + I]) == center
            value = x[i, j, k]
            if value >= thresh
		push!(maxs_array, x[i, j, k])
		push!(points, (i, j, k))
            end
        end
    end
    if normalize && maxs_array != []
        largest = maximum(maxs_array)
        smallest = minimum(maxs_array)
#	println("smallest: ", smallest)
	maxs_array = Float64.(maxs_array - smallest)/Float64(largest-smallest)
    end
    for k = 1:length(points[:,1])
        maxs[points[k]] = maxs_array[k]
    end
    maxs
end

function getlocalmaxs(pyr::Array{Any, 1}, levels::Int64, threshes::Array{Float64,1}, normalize::Bool = true)
    if normalize == true
        println("Maxs will be normalized")
    else
        println("Maxs will be nonnormalized")
    end
    maxs = []
    for l = 1:levels
        for ((i, j, k), value) = arglocalmax(pyr[l], 2, threshes[l], normalize)
            push!(maxs, (i, j, k, value, l))
        end
    end
    ordering = tup -> (tup[4], tup[5])
    sort!(maxs, by=ordering, rev=true)
    println("Local maxs found and sorted by value and level, for all levels of Pyramid")
    maxs
end

function makelappyr{T}(vol::Array{T, 3}, ord::Int=1)
    println("Creating Laplace Pyramid")
    L = Int(ceil(log2(minimum(size(vol)))) + 1)
    pyr = Any[vol]
    for l = 2:L
        if any(m -> m <= ord, size(pyr[l - 1]))
            break
        end
        push!(pyr, decvol(binomfilt3(pyr[l - 1], ord)))
        up = interpbinom(pyr[l], ord)
        pyr[l - 1] -= up[map(m -> 1:m, size(pyr[l - 1]))...]
    end
    println("Pyramid created")
    pyr
end

function normmaxs(pyr::Array{Any, 1})

    # Normalizes the maxs at each level of the pyramid.
    
    numlevels = length(pyr)
    for i = 1:numlevels
        if j == 1
            vol = pyr[i]
            vec = vol[:]
	    out = Float64.(vol - minimum(vec))/Float64(maximum(vec) - minimum(vec))
            pyr[i] = out
	end
    end
    println("Pyramid normalized")
    pyr
end


#---------------------------------------------------------------------------
ord_1 = 50

vol_1 = getvol(Int(1))

pyr_1 = makelappyr(vol_1, ord_1)

levels_1 = size(pyr_1)[1]

threshes_1 = Float64.([1808.38, 383.3, 600, 0, 0, 0, 0])

loc_max_1 = getlocalmaxs(pyr_1, levels_1, threshes_1, true)

#------------------Blob Detection and Trimming-------------------------------

function blob_upscale(loc_max::Array{Any,1}, ord::Int = 1)

#   Scales the coordinates to the base level based on their level in the pyramid.

    sig_sq = (ord * (1/4)) + (1/4)
    sig = sqrt(sig_sq)
    
    scaled_pts = []
    for i = 1:length(loc_max)
        level = loc_max[i][5]
	scale = 2^(level-1)
	scaled_tup = (scale*loc_max[i][1], scale*loc_max[i][2], scale*loc_max[i][3], loc_max[i][4], loc_max[i][5], scale*sig)
	push!(scaled_pts, scaled_tup)
    end
    println("Blob sizes have been scaled appropriately")
    scaled_pts
end

#---------------------------------------------------------------------------

loc_max_new = blob_upscale(loc_max_1, ord_1)

#--------------------------------------------------------------------------


function centerdist3{T}(sphere1::Array{T, 1}, sphere2::Array{T, 1})

#   Determines the distance between the center of 2 spheres.

    assert(size(sphere1)[1] == 3)
    assert(size(sphere2)[1] == 3)

    x1 = sphere1[1]
    y1 = sphere1[2]
    z1 = sphere1[3]

    x2 = sphere2[1]
    y2 = sphere2[2]
    z2 = sphere2[3]

    dist1 = sqrt((x1-x2)^2+(y1-y2)^2)
    dist2 = sqrt(dist1^2 + (z1-z2)^2)

    dist2
end


function blob_overlap{T}(blob1::Array{T,1}, sig1::Float64, blob2::Array{T,1}, sig2::Float64)

#   Determines the amount of overlap between two 3D blobs. Assume that the order of each inmput array is (x, y, z, sig)'''

    assert(size(blob1)[1] == 3)
    assert(size(blob2)[1] == 3)

    r1 = sig1*sqrt(2)  # radius of blob1 is sig1*sqrt(2)
    r2 = sig2*sqrt(2)  # radius of blob2 is sig2*sqrt(2)

    blob_coord1 = blob1[1:3]
    blob_coord2 = blob2[1:3]

    d = centerdist3(blob_coord1, blob_coord2)

    if d > r1 + r2
        return 0
    end

    if d <= abs(r1 - r2)
        return 1
    end

    
    dummy1 = (r1 + r2 -d)^2
    dummy2 = (d^2) + 2*d*r2 - 3*(r2^2) + 2*d*r1 + 6*r2*r1 - 3*(r1^2)
    dummy3 = 12*d
    intersect_vol = (pi*dummy1*dummy2)/dummy3

    radii = [r1, r2]
    radius = minimum(radii)
    volume = (4/3)*pi*(radius^3)

    intersect_vol/volume
end


function blob_trimmer(x::Array{Any,1},  threshold = 0.2)

    # Determines the amount of overlap between two blobs. If the overlap ratio is too big, the blob that has the lower local max value is deleted. 

    println("Blobs are being trimmed")
    allblobcombs = collect(combinations(x,2))
    for comb in allblobcombs
        coord1 = [comb[1][1], comb[1][2], comb[1][3]]
	coord2 = [comb[2][1], comb[2][2], comb[2][3]]
	over = blob_overlap(coord1, comb[1][6], coord2, comb[2][6])
        if over > threshold
	    if comb[1][4] > comb[2][4]
	        x = filter(y -> y != comb[2], x)
            else
	        x = filter(y -> y != comb[1], x)
            end
        end
    end
    println("Trimming finished")
    println("Size of blobs list after trimming: ", size(x)[1])
    x
end


#-------------------------------------------------------------------------

blobs_trimmed = blob_trimmer(loc_max_new)

#-------------------------------------------------------------------------

function upsample(pyr::Array{Any, 1}, l, numupsamples, ord::Int)

    # upsamples a volume at a specific level of the pyramid a specific number of times

    currentvol = pyr[l]
    for k = 1:numupsamples
        currentvol = interpbinom(currentvol, ord)
    end
    currentvol
end

function get_locvals_upperlevels(pyr::Array{Any, 1}, numlevels::Int64, locmaxinfo::Tuple{Int64, Int64, Int64, Float64, Int64, Float64}, ord::Int)

    # Takes in a local max for a specific level and finds the corresponding values for this local max at levels above it. Then it returns the values for levels above it, along with their corresponding sigma values.

    l = locmaxinfo[5]
    numupperlevels = numlevels - l
    upperlevelmaxs = []
    vol = pyr[l]
    currentsig = locmaxinfo[6]
    coors = [locmaxinfo[1], locmaxinfo[2], locmaxinfo[3]]./(2^(l-1))

    for k = 1:numupperlevels
        scale = k
        upsampledvol = upsample(pyr, locmaxinfo[5] + k, k, ord)
	vallocmax = upsampledvol[Int(coors[1]), Int(coors[2]), Int(coors[3])]
	newsig = currentsig*(2^scale)
	sig_tup = (newsig, vallocmax)
	push!(upperlevelmaxs, sig_tup)
    end
    upperlevelmaxs
end


function get_locvals_lowerlevels(pyr::Array{Any, 1}, numlevel::Int64, locmaxinfo::Tuple{Int64, Int64, Int64, Float64, Int64, Float64}, ord::Int)

    # Takes in a local max for a specfic level and finds the corresponding values for this local max at levels below it. Then it returns the values for the levels below it, along with their corresponding sigma values.

    l = locmaxinfo[5]
    numlowerlevels = l - 1
    lowerlevelmaxs = []
    vol = pyr[l]
    currentsig = locmaxinfo[6]
    coors = [locmaxinfo[1], locmaxinfo[2], locmaxinfo[3]]./(2^(l-1))

    for k = 1:numlowerlevels
        currentvol = pyr[k]
        scale = l-k
	newcoors = coors.*(2^scale)
	vallocmax = currentvol[Int(newcoors[1]), Int(newcoors[2]), Int(newcoors[3])]
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


function continuous_sig(pyr::Array{Any, 1}, loc_maxs::Array{Any, 1}, ord::Int = 1)

    # Uses interpolation to create a continuous function out or the scale parameter to determine the appropriate size of a blob a blob for a given local max.

    new_blobs_trimmed = []
    numlocmaxs = length(loc_maxs)
    levels = length(pyr)
    sig_sq = (ord * (1/4))+(1/4)
    sig = sqrt(sig_sq)    # blob size for base of the pyramid
    plotticks = linspace(1, levels, 350)
    plotticks = Array(plotticks) 

    for k = 1:numlocmaxs
        currentmax = loc_maxs[k]
	upperlevels = get_locvals_upperlevels(pyr, levels, currentmax, Int(ord))
	lowerlevels = get_locvals_lowerlevels(pyr, levels, currentmax, Int(ord))
	levelmaxs = []
	
	for j = 1:length(lowerlevels)
	    push!(levelmaxs, lowerlevels[j])
        end

	push!(levelmaxs, (currentmax[6], currentmax[4]))

        for j = 1:length(upperlevels)
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
        push!(new_blobs_trimmed, (loc_maxs[k][1], loc_maxs[k][2], loc_maxs[k][3], val, maxpoint, sig*2^(maxpoint-1)))
        println("updated sigma ", k)
    end

    println("Finished creating a continuous scale parameter")
    new_blobs_trimmed

end

#-------------------------------------------------------------------------------

#new_blobs_trimmed = continuous_sig(pyr_1, blobs_trimmed, ord_1)
#for i = 1:length(new_trimmed_blobs)
#    println(new_trimmed_blobs[i])
#end


#-----------------------3D VOLUME PLOT------------------------------------------
cont3d = mlab.contour3d(vol_1) # <-- If you only what to see the blobs
mlab.axes(cont3d) 
blob_matrix = Matrix(0,4)
for i = 1:length(blobs_trimmed)
    blob_info = [blobs_trimmed[i][1], blobs_trimmed[i][2], blobs_trimmed[i][3], blobs_trimmed[i][6]]'
    blob_matrix = cat(1, blob_matrix, blob_info)
end

for i = 1:length(blob_matrix[:, 1])
    pts = mlab.points3d(blob_matrix[i,1], blob_matrix[i,2], blob_matrix[i,3], scale_factor = 2*blob_matrix[i,4]*sqrt(2), color = (0,0,0))
end

#mlab.axes(extent = [0, 224, 0, 224, 0, 161])

#pts = mlab.points3d(blob_matrix[:,1], blob_matrix[:,2], blob_matrix[:,3], scale_#factor = 2*blob_matrix[:,4]*sqrt(2), color= (0,0,0))

#mayavi.mlab.pipeline[:volume](mlab.pipeline[:scalar_field](vol_1))
mlab.show()


#------------2D plots, using imshow, with detected blobs circled--------------

#img = vol_1[:,:,47]

#fig = figure()
#ax = gca(projection = "3d")
#view(vol_1)

#fig, ax = subplots(1)
#imshow(img)
#ax = gca()
#blob_coors = []
#for blob in blobs_trimmed
#    coorstup = (blob[1], blob[2], blob[3])
#    push!(blob_coors,coorstup)
#    c = circle((blob[2], blob[1]), 2, edgecolor = "black", facecolor = "none")
#    c[:radius] = sqrt(2)*blob[6]
#    ax[:add_patch](c)
#end
#show()
#print(blob_coors)
