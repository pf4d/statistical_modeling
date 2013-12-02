import sys
sys.path.append('../')

from src.regstats         import *
from pylab                import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.io             import loadmat
from scipy.optimize       import fmin
from numpy.random         import choice

#---------------------------------------------------------------------
# function to be fit :
def f(t, p):
  alpha = p[0]
  beta  = p[1]
  delta = p[2]
  yhat  = alpha - beta*exp(-delta*t)
  return yhat


def bootOut(betaBoot, bhat, tits, alpha, fn):
  """
  formulate statistics and plot output of bootstrap.
  """
  bm    = mean(betaBoot, axis=0)
  bias  = bm - bhat
  bse   = std(betaBoot, axis=0)
  p     = len(bhat)
  B     = len(betaBoot[:,0])
  nbins = max(10, round(B/50.0))
  
  fig = figure(figsize=(12,5))
  for i in range(p):
    ax = fig.add_subplot(100 + p*10 + 1 + i)
    ax.hist(betaBoot[:,i], nbins)
    ax.set_xlabel('Bootstrap Estimates')
    ax.set_ylabel('Frequency')
    ax.set_title('Bootstrap ' + tits[i] + ' Estimates')
    ax.grid()
  tight_layout()
  #savefig(fn, dpi=300)
  show()
  
  sbeta = betaBoot.copy()
  for i in range(p):
    sbeta[:,i] = sort(sbeta[:,i])
  
  cilow = sbeta[round(alpha*(B-1)/2.0), :]
  cihi  = sbeta[round(B-alpha*(B-1)/2.0 - 1), :]
  
  out   = {'true'  : bhat,
           'mean'  : bm,
           'se'    : bse,
           'bias'  : bias,
           'cilow' : cilow,
           'cihi'  : cihi}
  return out



# ============================================================== #
# Computes the parameter estimate standard errors using          #
# bootstrapping of data pairs, for a 3-parameter growth model.   #
# ============================================================== #
# data:
data = loadmat('../data/growth.mat')   
x    = data['growth'][0][0][0].T[0]
y    = data['growth'][0][0][1].T[0]

# initial conditions :
beta0 = [1, 1, 1]

conf = 0.95
out  = nonlinRegstats(x, y, f, beta0, conf)

n    = len(x)
p    = len(beta0)
J    = out['fjac'].T
MSE  = out['MSE']
bhat = out['bhat']

JTJ  = dot(J.T, J)
SED  = sqrt(diag(inv(JTJ) * MSE))

# plot the data :
plot(x, y,         'ro')
plot(x, f(x,bhat), 'k-', lw=2.0)
grid()
xlabel(r'$x$')
ylabel(r'$y$')
tight_layout()
#savefig('growth.png', dpi=300)
show()


# ================================================================ #
# Performs Bootstrap Method 2 (resampling residuals) to estimate   #
# standard errors for the 4-parameter generalized logisitic model. #
# ================================================================ #
B       = 5000
yhat    = out['yhat']
resid   = out['resid']
lev     = diag(dot(J, dot(inv(JTJ), J.T)))
yhatMat = tensordot(ones(B), yhat, 0)

modres   = resid / sqrt(1 - lev)
modres  -= mean(modres)
residMat = choice(modres, B*n, replace=True)
residMat = reshape(residMat, (B,n))

bsamp = yhatMat + residMat
for i in range(B):
  out = nonlinRegstats(x, bsamp[i,:], f, bhat, conf)
  if i == 0:
    betaBoot2 = out['bhat']
  else:
    betaBoot2 = vstack((betaBoot2, out['bhat']))

alpha = 0.05
tits  = [r'$\alpha$', r'$\beta$', r'$\delta$']
out2  = bootOut(betaBoot2, bhat, tits, alpha, 'meth2.png')


# ================================================================ #
# Performs Bootstrap Method 1 (resampling (x,y) pairs) to estimate #
# standard errors for the 4-parameter generalized logistic model.  #
# ================================================================ #
for i in range(B):
  bsamp = choice(arange(n), n, replace=True)
  xboot = x[bsamp]
  yboot = y[bsamp]
  out = nonlinRegstats(xboot, yboot, f, bhat, conf)
  if i == 0:
    betaBoot1 = out['bhat']
  else:
    betaBoot1 = vstack((betaBoot1, out['bhat']))

out1 = bootOut(betaBoot1, bhat, tits, alpha, 'meth1.png')




