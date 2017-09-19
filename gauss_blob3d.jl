# Create Gaussian Filter

using PyPlot

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
    kpad1[Int(M-floor(length(k)/2)+1):M] = kpad1[1:Int(floor(length(k)/2))]
    fft!(kpad1)
    for n = 1:N, d = 1:D X[:, n, d].*= kpad1 end

    ifft!(X, 1)
    fft!(X, 2)

    kpad2[1:Int(ceil(length(k)/2))] = k[Int(ceil(length(k)/2)):Int(length(k))]
    kpad2[Int(N-floor(length(k)/2)+1):N] = kpad2[1:Int(floor(length(k)/2))]
    fft!(kpad2)
    for m = 1:M, d = 1:D X[m, :, d].*= kpad2 end

    ifft!(X, 2)
    fft!(X, 3)

    kpad3[1:Int(ceil(length(k)/2))] = k[Int(ceil(length(k)/2)):Int(length(k))]
    kpad3[Int(D-floor(length(k)/2)+1):D] = kpad3[1:Int(floor(length(k)/2))]
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

function arglocalmax{T}(x::Array{T, 3}, r=1, thresh=0)
    m1, m2, m3 = size(x)
    I = -r:r
    i0 = r + 1
    center = (i0, i0, i0)
    maxs = Dict{Tuple{Int, Int, Int}, T}()
    for i = 2:m1-1, j = 2:m2-1, k = 2:m3-1
        if argmax(x[i + I, j + I, k + I]) == center
            value = x[i, j, k]
            if value >= thresh
                maxs[(i, j, k)] = x[i, j, k]
            end
        end
    end
    maxs
end

function getlocalmaxs{T}(pyr::Array{Array{T, 3},1}, level::Int64, thresh=0)
    maxs = []
    for l = levels
        for ((i, j, k), value) = arglocalmax(pyr[l], 1, thresh)
            push!(maxs, (i, j, k, value, l))
        end
    end
    ordering = tup -> (tup[4], tup[5])
    sort!(maxs, by=ordering, rev=true)
    maxs
end

function makelappyr{T}(vol::Array{T, 3}, pts, i, sig, ord::Int=1)
    L = Int(ceil(log2(minimum(size(vol)))) + 1)
    pyr = Array[vol]
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

vol_1 = getvol(Int(1))

vol_1 = squeeze(vol_1, 3)

pyr_1 = makelappyr(vol_1, 1, 6, 1);

levels_1 = size(pyr_1)[1];

loc_max = getlocalmaxs(pyr_1, levels_1)
