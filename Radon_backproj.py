import time

def interp(x, y, ndimphan, w):
    import math

    tstep = 2.0 / (ndimphan - 1.0)

    rx = (x + 1.0) / tstep + 1.0
    ry = (y + 1.0) / tstep + 1.0
    nx = math.floor(rx)
    ny = math.floor(ry)

    ax1 = rx - nx
    ay1 = ry - ny
    ax0 = 1.0 - ax1
    ay0 = 1.0 - ay1

    return ax0*(ay0*w[nx-1, ny-1] + ay1*w[nx-1, ny]) + ax1*(ay0*w[nx, ny-1] + ay1*w[nx, ny])


def Radon(phan, nproj, nlines): # Rproj nlines x nproj
    from PIL import Image
    import numpy as np
    
    # opens phantom image, converts image to numpy arrays
    im = Image.open(phan)
    imarray = np.array(im)
    
    ny, nx = imarray.shape #switches values of dimensions; y-axis first, x-axis second
    ndimphan = ny
    stepxy = 2.0 / (ndimphan - 1.0)
    tstep = stepxy
    tstepsmall = tstep * 0.25

    angle = np.linspace(0, np.pi, nproj+1)
    angle = angle[:nproj]

    co = np.cos(angle)
    si = np.sin(angle)

    s = np.linspace(-1.0, 1.0, nlines)
    halfchord = np.sqrt(1.0 - np.multiply(s, s))

    s = np.expand_dims(s, axis=-1)
    co1 = np.expand_dims(co, axis=0)
    si1 = np.expand_dims(si, axis=0)
    sco = np.matmul(s, co1)
    ssi = np.matmul(s, si1)
    Nt = np.int_(2.0 * halfchord / tstepsmall)

    Rproj = np.zeros((nlines, nproj))
    extphan = np.pad(imarray,1) #to pad the image

    # for each \omega
    for npr in range(nproj):
        proj = np.zeros(nlines)

        # for each line perpendicular to \omaga
        for il in range(1, nlines-1):
            total = 0.0

            # integrating along the line by using interpolation
            for it in range(Nt[il]+1):
                t = -halfchord[il] + it * tstepsmall
                x = t * si[npr] + sco[il, npr]
                y = -t * co[npr] + ssi[il, npr]
                v = interp(x,y,ndimphan,extphan) #swtichted interp_tf function interp to compute values
                total = total + v
            proj[il] = total * tstepsmall
        Rproj[:, npr] = proj
        resultPic = Image.fromarray(Rproj)
        resultPic.save('output.tif')


def backproj(Rproj, ndimphan):
    from PIL import Image
    import numpy as np
    import math

    im = Image.open(Rproj)
    imarray = np.array(im)

    nproj, nlines = imarray.shape

    sstep = 2.0/(nlines-1)
    anglestep = math.pi/nproj
    x = np.linspace(-1.0, 1.0, ndimphan)
    y = np.linspace(-1.0, 1.0, ndimphan)
    angle = np.linspace(0., np.pi, nproj+1)
    angle = angle[:nproj]
    co = np.cos(angle)
    si = np.sin(angle)

    reconst = np.zeros((ndimphan, ndimphan))
    weight = np.ones(nproj)
    # weight in trapezoidal rule
    weight[0] = 0.5
    weight[nproj-1] = 0.5
    weight = anglestep * weight

    for ix in range(ndimphan):
        for iy in range(ndimphan):
            if x[ix]*x[ix]+y[iy]*y[iy] < 1.0:

                # integrating over all angles \omega
                for npr in range(nproj):
                    s = x[ix] * co[npr] + y[iy] * si[npr]
                    position = (nlines + 1) / 2 + s / sstep
                    nline0 = math.floor(position)
                    nline1 = nline0 + 1
                    alpha1 = position - nline0
                    alpha0 = 1.0 - alpha1
                    value = alpha0 * imarray[nline0 - 1, npr] + alpha1 * imarray[nline1 - 1, npr]
                    reconst[ix, iy] = reconst[ix, iy] + value*weight[npr]/(2*math.pi)
    resultPic = Image.fromarray(reconst)
    resultPic.save('Routput.tif')

from PIL import Image

first = time.perf_counter()
Radon("Sheep_Logan.tif", 100, 100)
second = time.perf_counter()
print(f"Seconds: {second - first:0.4f}")

first = time.perf_counter()
backproj('output.tif', 99)
second = time.perf_counter()

print(f"Seconds (Back Projection): {second - first:0.4f}")

