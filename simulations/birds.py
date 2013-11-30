import sys
src_directory = '../'
sys.path.append(src_directory)

from pylab        import *
from scipy.io     import loadmat
from src.regstats import *

data = loadmat('../data/extinct.mat')['extinct'][0][0]

srt = argsort(data[2].flatten()) # x-value increasing only.

species  = data[0].flatten()[srt]
tof      = data[1].flatten()[srt]
numpairs = data[2].flatten()[srt]
avgsize  = data[3].flatten()[srt]
mig_stat = data[4].flatten()[srt]

L = where(avgsize == 'L')[0]
S = where(avgsize == 'S')[0]
M = where(mig_stat == 'M')[0]
R = where(mig_stat == 'R')[0]

LM = intersect1d(L,M)
LR = intersect1d(L,R)
SM = intersect1d(S,M)
SR = intersect1d(S,R)

n = len(tof)

x1 = log(numpairs)
x2 = (avgsize == 'L').astype(float)
x3 = (mig_stat == 'M').astype(float)
x4 = x1 * x2

X = array([x1, x2, x3, x4])
y = log(tof)

out = linRegstats(X, y, 0.95)

bhat = out['bhat']
yhat = out['yhat']

ciy  = out['CIY']

ltred = '#ff8989'
ltgry = '#878787'

plot(x1[LM], y[LM],      'ko',         label=r'LM')
plot(x1[LR], y[LR],      'ro',         label=r'LR')
plot(x1[SM], y[SM],      'gs',         label=r'SM')
plot(x1[SR], y[SR],      'ys',         label=r'SR')

plot(x1[LM], yhat[LM],   'k-', lw=2.0, label=r'LM $\vec{x}_{LS}$')
plot(x1[LR], yhat[LR],   'r-', lw=2.0, label=r'LR $\vec{x}_{LS}$')
plot(x1[SM], yhat[SM],   'g-', lw=2.0, label=r'SM $\vec{x}_{LS}$')
plot(x1[SR], yhat[SR],   'y-', lw=2.0, label=r'SR $\vec{x}_{LS}$')

#plot(x1[LM], ciy[0][LM], 'k:', lw=3.0, label=r'LM C.I.')
#plot(x1[LM], ciy[1][LM], 'k:', lw=3.0)
#plot(x1[LR], ciy[0][LR], 'r:', lw=3.0, label=r'LR C.I.')
#plot(x1[LR], ciy[1][LR], 'r:', lw=3.0)
#plot(x1[SM], ciy[0][SM], 'g:', lw=3.0, label=r'SM C.I.')
#plot(x1[SM], ciy[1][SM], 'g:', lw=3.0)
#plot(x1[SR], ciy[0][SR], 'y:', lw=3.0, label=r'SR C.I.')
#plot(x1[SR], ciy[1][SR], 'y:', lw=3.0)

xlabel('log of nesting pairs') 
ylabel('log of average extinction time')
title('Nesting Pairs vs. Extinction Time')
grid()
leg = legend(loc='upper left')
leg.get_frame().set_alpha(0.5)
tight_layout()
#savefig('../doc/images/prb3a.png', dpi=300)
show()


X = array([x1, x2, x3])
y = log(tof)

out = linRegstats(X, y, 0.95)

bhat = out['bhat']
yhat = out['yhat']

ciy  = out['CIY']

ltred = '#ff8989'
ltgry = '#878787'

plot(x1[LM], y[LM],      'ko',         label=r'LM')
plot(x1[LR], y[LR],      'ro',         label=r'LR')
plot(x1[SM], y[SM],      'gs',         label=r'SM')
plot(x1[SR], y[SR],      'ys',         label=r'SR')

plot(x1[LM], yhat[LM],   'k-', lw=2.0, label=r'LM $\vec{x}_{LS}$')
plot(x1[LR], yhat[LR],   'r-', lw=2.0, label=r'LR $\vec{x}_{LS}$')
plot(x1[SM], yhat[SM],   'g-', lw=2.0, label=r'SM $\vec{x}_{LS}$')
plot(x1[SR], yhat[SR],   'y-', lw=2.0, label=r'SR $\vec{x}_{LS}$')

#plot(x1[LM], ciy[0][LM], 'k:', lw=3.0, label=r'LM C.I.')
#plot(x1[LM], ciy[1][LM], 'k:', lw=3.0)
plot(x1[LR], ciy[0][LR], 'r:', lw=3.0, label=r'LR C.I.')
plot(x1[LR], ciy[1][LR], 'r:', lw=3.0)
plot(x1[SM], ciy[0][SM], 'g:', lw=3.0, label=r'SM C.I.')
plot(x1[SM], ciy[1][SM], 'g:', lw=3.0)
#plot(x1[SR], ciy[0][SR], 'y:', lw=3.0, label=r'SR C.I.')
#plot(x1[SR], ciy[1][SR], 'y:', lw=3.0)

xlabel('log of nesting pairs') 
ylabel('log of average extinction time')
title('Nesting Pairs vs. Extinction Time')
grid()
leg = legend(loc='upper left')
leg.get_frame().set_alpha(0.5)
tight_layout()
#savefig('../doc/images/prb3b.png', dpi=300)
show()


