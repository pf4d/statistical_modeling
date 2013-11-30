from pylab       import *
from scipy.io    import loadmat
from scipy.stats import scoreatpercentile

def iqr(arr):
  arr           = sort(arr.copy())
  upperQuartile = scoreatpercentile(arr,.75)
  lowerQuartile = scoreatpercentile(arr,.25)
  iqr           = upperQuartile - lowerQuartile
  return iqr

data = loadmat('../data/irondiet.mat')

fe3  = data['irondiet']['fe3'][0][0][0]
fe4  = data['irondiet']['fe4'][0][0][0]

fig = figure(figsize=(12,5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

tits = [r'Fe$^{3+}$', r'Fe$^{4+}$']
axs  = [ax1, ax2]
fes  = [fe3, fe4]

for ax, tit, fe in zip(axs, tits, fes):
  ax.hist(fe, 10)
  ax.set_xlabel(r'Percent remaining')
  ax.set_ylabel(r'Number of occurances')
  ax.set_ylim([0,5])
  ax.set_xlim([0,14])
  ax.set_title(tit)
  ax.grid()

  mu      = mean(fe)
  med     = median(fe)
  sigma   = std(fe)
  fe_iqr  = iqr(fe)
  v_m_rat = sigma**2 / mu
  textstr =   '$\mu = %.2f$\n' \
            + '$\mathrm{median} = %.2f$\n' \
            + '$\sigma = %.2f$\n' \
            + '$\mathrm{IQR} = %.3f$\n' \
            + '$\sigma^2 / \mu = %.2f$'
  textstr = textstr % (mu, med, sigma, fe_iqr, v_m_rat)
  
  # these are matplotlib.patch.Patch properties
  props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
  
  # place a text box in upper left in axes coords
  ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=14,
          verticalalignment='top', bbox=props)

#savefig('../doc/images/prb3.png', dpi=300)
show()

