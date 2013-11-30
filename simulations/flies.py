import sys
src_directory = '../'
sys.path.append(src_directory)

from pylab          import *
from scipy.optimize import leastsq
from scipy.io       import loadmat
from src.regstats   import linRegstats, prbplotObj
from scipy.stats    import probplot

# ==================================================================== #
# Problem 1: Scatterplot and fit of wing size vs. latitude for 2 flies #
# ==================================================================== #
data   = loadmat('../data/evolution.mat')
lat    = data['evol']['latitude'][0][0].T[0]
wing   = data['evol']['wingsize'][0][0].T[0]
cont   = data['evol']['cont'][0][0].T[0]

nam = where(cont == 'NAM')
eur = where(cont == 'EUR')
n   = len(lat)

x1  = lat
x2  = (cont == 'EUR').astype(float)
x3  = x1 * x2

X  = array([x1, x2, x3])
y  = wing

out  = linRegstats(X, y, 0.95)

bhat = out['bhat']
yhat = out['yhat']

ciy  = out['CIY']

plot(lat[nam], wing[nam],   'ko', label=r'NAM')
plot(lat[eur], wing[eur],   'ro', label=r'EUR')
plot(lat[nam], yhat[nam],   'k-', lw=2.0, label=r'NAM $\vec{x}_{LS}$')
plot(lat[eur], yhat[eur],   'r-', lw=2.0, label=r'EUR $\vec{x}_{LS}$')
plot(lat[nam], ciy[0][nam], 'k:', lw=3.0, label=r'NAM C.I.')
plot(lat[nam], ciy[1][nam], 'k:', lw=3.0)
plot(lat[eur], ciy[0][eur], 'r:', lw=3.0, label=r'EUR C.I.')
plot(lat[eur], ciy[1][eur], 'r:', lw=3.0)
xlabel('Latitude') 
ylabel('Average Wing Size (log mm)')
title('Wing Size vs. Latitude for D. subobscura Flies')
grid()
leg = legend(loc='lower right')
leg.get_frame().set_alpha(0.5)
#savefig('../doc/images/prb1a.png', dpi=300)
show()



