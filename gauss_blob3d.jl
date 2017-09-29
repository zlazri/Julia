# GAUSSIAN FILTER FOR LAPLACIAN PYRAMID. BLOB DETECTOR

# TODO:
# (1)  USE BLOB DETECTION TO PLOT 3D SPHERES AROUND DETECTED BLOBS
# (2)  MAKE IT SO THAT YOU CAN RECOVER BLOB SIZES NOT OF THE FORM IN SAM'S EMAIL (SEE EMIAL) 
# (3)  GO BACK AND COMMENT SO IT'S EASY FOR OTHERS TO READ


using ImageView
using ImageView
using PyPlot
using PyCall
@pyimport matplotlib.patches as patch
circle = patch.pymember("Circle")


f = open("/home/zlazri/Documents/Vol_Data/Image_0001_0001.raw")

width = 224
height = 224
nslices = 161
nchans = 4
tsteps = 50
m = Mmap.mmap(f, Array{UInt16, 5}, (width, height, nchans, nslices, tsteps))

chan = 4

getvol(tstep::Int) = m[:, :, chan, 1:nslices, tstep]

function getgausskern(pts, i, sig)                   # pts = scales # of points in interval, int = size interval, sig = scale parm

    assert(pts > 0)
    assert(i > 0)
    assert(sig > 0)
    
    t = sig^2
    x = linspace(-i, i, (i*2+1)*pts)                 # t*2+1 is number of integer points in interval (+1 at end accts for 0)
    y = (e.^(-(x.^2)./(2*t)))./(t*sqrt(2*pi))        # Gaussian function centered at x=0
end

function gaussfilt3{T}(x::Array{T, 3}, pts, i, sig)
    k = getgausskern(pts, i, sig)

    M, N, D = size(x)

    X = fft(x, 1)

    kpad1 = zeros(Complex128, (M, 1, 1))
    kpad2 = zeros(Complex128, (1, N, 1))
    kpad3 = zeros(Complex128, (1, 1, D))
    
    kpad1[1:Int(ceil(length(k)/2))] = k[Int(ceil(length(k)/2)):Int(length(k))]
    kpad1[Int(M-floor(length(k)/2)+1):M] = k[1:Int(floor(length(k)/2))]
    fft!(kpad1)
    for n = 1:N, d = 1:D X[:, n, d].*= kpad1 end

    ifft!(X, 1)
    fft!(X, 2)

    kpad2[1:Int(ceil(length(k)/2))] = k[Int(ceil(length(k)/2)):Int(length(k))]
    kpad2[Int(N-floor(length(k)/2)+1):N] = k[1:Int(floor(length(k)/2))]
    fft!(kpad2)
    for m = 1:M, d = 1:D X[m, :, d].*= kpad2 end

    ifft!(X, 2)
    fft!(X, 3)

    kpad3[1:Int(ceil(length(k)/2))] = k[Int(ceil(length(k)/2)):Int(length(k))]
    kpad3[Int(D-floor(length(k)/2)+1):D] = k[1:Int(floor(length(k)/2))]
    fft!(kpad3)
    for m = 1:M, n = 1:N X[m, n, :].*= kpad3 end

    real(ifft(X, 3))
end

decvol{T}(x::Array{T, 3}) = x[map(m -> 1:2:m, size(x))...]

function interpvol{T}(x::Array{T, 3}, interpfilt)
    y = zeros(T, map(x -> 2*x, size(x)))
    y[map(m -> 1:2:2m, size(x))...] = x
    interpfilt(y)
end

function interpgauss{T}(x::Array{T, 3}, pts, i, sig)
    interpvol(x, y -> gaussfilt3(y, pts, i, sig))
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

function arglocalmax{T}(x::Array{T, 3}, r=1, thresh = 900)
    m1, m2, m3 = size(x)
    I = -r:r
    i0 = r + 1
    center = (i0, i0, i0)
    maxs = Dict{Tuple{Int, Int, Int}, T}()
    for i = (r+1):(m1-r-1), j = (r+1):(m2-r-1), k = (r+1):(m3-r-1)
        if argmax(x[i + I, j + I, k + I]) == center
            value = x[i, j, k]
            if value >= thresh
                maxs[(i, j, k)] = x[i, j, k]
            end
        end
    end
    maxs
end

function getlocalmaxs{T}(pyr::Array{Array{T, 3},1}, levels::Int64, sig::Int64, threshes::Array{Int64,1})
    maxs = []
    for l = 1:levels
        for ((i, j, k), value) = arglocalmax(pyr[l], 2, threshes[l])
            push!(maxs, (i, j, k, value, l, sig))
        end
    end
    ordering = tup -> (tup[4], tup[5])
    sort!(maxs, by=ordering, rev=true)
    maxs
end

function makelappyr{T}(vol::Array{T, 3}, pts, i, sig)
    L = Int(ceil(log2(minimum(size(vol)))) + 1)
    pyr = Array(Array{Float64, 3}, 1)
    fill!(pyr, vol)
    for l = 2:L
	M, N, D = size(pyr[l - 1])
        if M > (i*2*pts) && N > (i*2*pts) && D > (i*2*pts)
            push!(pyr, decvol(gaussfilt3(pyr[l - 1], pts, i, sig)))
            up = interpgauss(pyr[l], pts, i, sig)
            pyr[l - 1] -= up[map(m -> 1:m, size(pyr[l - 1]))...]
	else
	    break
	end
    end
    pyr
end


#---------------------------------------------------------------------------
pts_1 = 5

i_1 = 6

sig_1 = 5

vol_1 = getvol(Int(1))

vol_1 = squeeze(vol_1, 3)

pyr_1 = makelappyr(vol_1, pts_1, i_1, sig_1)

levels_1 = size(pyr_1)[1]

threshes_1 = [2000, 100, 100]

loc_max_1 = getlocalmaxs(pyr_1, levels_1, sig_1, threshes_1)
#------------------------------------------------------------------------------

function blob_upscale(loc_max::Array{Any,1})

#   Scales the coordinates to the base level based on their level in the pyramid.

    scaled_pts = []
    for i = 1:length(loc_max)
        level = loc_max[i][5]
	scale = 2^(level-1)
	scaled_tup = (scale*loc_max[i][1], scale*loc_max[i][2], scale*loc_max[i][3], loc_max[i][4], loc_max[i][5], scale*loc_max[i][6])
	push!(scaled_pts, scaled_tup)
    end
    scaled_pts
end

#---------------------------------------------------------------------------

loc_max_new = blob_upscale(loc_max_1)

#--------------------------------------------------------------------------


function centerdist3{T}(sphere1::Array{T,1}, sphere2::Array{T,1})

#   Determines the distance between the center of 2 spheres

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


function blob_overlap{T}(blob1::Array{T,1}, sig1::Int64, blob2::Array{T,1}, sig2::Int64)

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

    allblobcombs = combinations(x,2)
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
    x
end


#-------------------------------------------------------------------------

blobs_trimmed = blob_trimmer(loc_max_new)


#3D VOLUME PLOT

@pyimport mayavi.mlab as mlab

cont3d = mlab.contour3d(vol_1) # <-- If you only what to see the blobs
mlab.axes(cont3d) 

blob_matrix = Matrix(0,4)
for i = 1:length(blobs_trimmed)
    blob_info = [blobs_trimmed[i][1], blobs_trimmed[i][2], blobs_trimmed[i][3], blobs_trimmed[i][6]]'
    blob_matrix = cat(1, blob_matrix, blob_info)
end

#pts = mlab.points3d(blob_matrix[:,1], blob_matrix[:,2], blob_matrix[:,3], blob_matrix[:,4]*sqrt(2), scale_mode = "none",  color= (0,0,0))

#mlab.pipeline[:volume](mlab.pipeline[:scalar_field](vol_1))
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
