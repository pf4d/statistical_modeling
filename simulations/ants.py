import sys
src_directory = '../'
sys.path.append(src_directory)

from pylab          import *
from scipy.stats    import scoreatpercentile, distributions, nbinom
from scipy.io       import loadmat
from scipy.optimize import minimize
from src.regstats   import negbinlike, chiTest

poisson = distributions.poisson.pmf

def iqr(arr):
  arr           = sort(arr.copy())
  upperQuartile = scoreatpercentile(arr, 75)
  lowerQuartile = scoreatpercentile(arr, 25)
  iqr           = upperQuartile - lowerQuartile
  return iqr

ants = loadmat('../data/hills.mat')['hills'][0]
num  = unique(ants)
freq = array([])
for v in num:
  freq = append(freq, len(where(ants == v)[0]))

mu         = mean(ants)                # mean
med        = median(ants)              # median
sigma      = std(ants)                 # standard deviation
fe_iqr     = iqr(ants)                 # IQR
v_m_rat    = sigma**2 / mu             # variance-to-mean ratio

#===============================================================================
# Poisson-distribution expected frequencies :
px         = poisson(num, mu)          # compute probabilities
ps_expfreq = px * len(ants)            # computes expected frequencies

#===============================================================================
# MLE estimation for negative-binomial distribution :

# Starting values for (r, p)
r0 = (mu + mu**2) / sigma**2
p0 = r0 / (mu + r0)

out = minimize(negbinlike, [r0, p0], args=(ants,), method='L-BFGS-B') 

r, p = out['x']                   # MLE

nbin       = nbinom(r, p)         # n-bin object
bx         = nbin.pmf(num)        # probabilities
nb_expfreq = bx * len(ants)       # expected frequency

#===============================================================================
# plotting :
fig      = figure()
ax       = fig.add_subplot(111)

ax.hist(ants, max(ants), color='0.4', histtype='stepfilled')
ax.plot(num + .5, nb_expfreq, 'ko', label='Neg-Bin expected freq')
ax.plot(num + .5, ps_expfreq, 'rs', label='Poisson expected freq')
ax.set_xlabel('# fireants per 50-meter square plot')
ax.set_ylabel('Frequency')
ax.set_title('Histogram of Fire-Ant Hill Counts')
ax.legend(loc='center right')
ax.grid()
textstr =   '$\mu = %.2f$\n' \
          + '$\mathrm{median} = %.2f$\n' \
          + '$\sigma = %.2f$\n' \
          + '$\mathrm{IQR} = %.3f$\n' \
          + '$\sigma^2 / \mu = %.2f$'
textstr = textstr % (mu, med, sigma, fe_iqr, v_m_rat)

# these are matplotlib.patch.Patch properties
props   = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# place a text box in upper left in axes coords
ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
#savefig('../doc/images/prb1a.png', dpi=300)
show()

#===============================================================================
# perform chi^2 test for Poisson distribution :
mle  = mean(ants)
px   = poisson(range(11), mle)            # for results < 11
pval = chiTest(ants, px, 11, 1)
print pval

#===============================================================================
# perform chi^2 test for negative binomial distribution :
px   = nbin.pmf(range(11))                # for results < 11
pval = chiTest(ants, px, 11, 2)
print pval

