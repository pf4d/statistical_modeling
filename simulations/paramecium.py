import sys
src_directory = '../'
sys.path.append(src_directory)

from pylab          import *
from scipy.optimize import leastsq
from scipy.io       import loadmat
from src.regstats   import nonlinRegstats, prbplotObj
from scipy.stats    import probplot

def weibull(t, beta):
  alpha = beta[0]
  sigma = beta[1]
  gamma = beta[2]
  y = alpha * (1 - exp(-(t/sigma)**gamma))
  return y

# =============================================== #
# Problem 3: Weibull fit to Growth Data with plot #
# =============================================== #
data   = loadmat('../data/paramecium.mat')     # Loads the growth data
growth = data['paramecium']['growth'][0][0][0] # Defines growth vector
time   = data['paramecium']['time'][0][0][0]   # Defines time vector
plot(time, growth, 'ko')                       # Plots growth vs. time
xlim([0,16])                                   # x-axis plotting limits
xlabel('Time (in days)') 
ylabel('Growth (orgs./ml)') 
title('Growth vs. Time') 
grid()
#savefig('../doc/images/prb3a.png', dpi=300)
show()

# ======================================================= #
# Problem 3c - Nonlinear Weibull model fit to growth data #
# ======================================================= #
beta0 = [11000, 4.18, 1.85]                 # Parameter starting values
conf  = 0.95                                # confidence region

# Vector of times from 0 to 16
time1 = arange(0, 16, 0.1)

# Performs nonlinear Weibull fit returning betahats, resids
out = nonlinRegstats(time, growth, weibull, beta0, conf)

# Computes Weibull predicted values (yhat's)
yhat  = out['yhat']
ci_y  = out['CIY']
ciy_u = out['CIY'][0]
ciy_l = out['CIY'][1]

# Plots 95% confidence bands
plot(time, growth, 'ko')
plot(time, yhat,   'r-',  lw=2.0, label=r'least squares')
plot(time, ciy_u,  'k--', lw=2.0, label=r'%.f%% CI for $\hat{y}$' % (100*conf))
plot(time, ciy_l,  'k--', lw=2.0)
xlim([0,16])
xlabel('Time (in days)') 
ylabel('Growth (orgs./ml)') 
title('Growth vs. Time') 
grid()
legend(loc='lower right')
#savefig('../doc/images/prb3c.png', dpi=300)
show()


# =========================================================== #
# Problem 3d - Residual and normal quantile plot of residuals #
# =========================================================== #
fig = figure(figsize=(12,5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

yhat_weib = growth - out['resid']
ax1.plot(yhat_weib, out['resid'], 'ko')
ax1.set_xlabel('Predicted Values') 
ax1.set_ylabel('Residuals') 
ax1.set_title('Residual Plot') 
ax1.grid()

# Normal quantile plot of residuals
p = prbplotObj(ax2)
probplot(out['resid'], plot=p)
ax2.set_xlabel('Standard Normal Quantiles') 
ax2.set_ylabel('Residuals') 
ax2.set_title('Normal Quantile Plot')
ax2.grid()
#savefig('../doc/images/prb3d.png', dpi=300)
show()

## ======================================================================= #
# Problem 3e,f - Individual 95% Confidence intervals for the 3 parameters #
# ======================================================================= #
#ci = nlparci(betahat1,resid1,J1)     # Computes CIs for (alpha,sigma,gamma)
#moe = (ci(:,2)-ci(:,1))/2            # Margin of error from CIs
#se = moe/tinv(.975,13)               # Standard errors from CIs

# Computes Weibull predicted values (bhat's)
bhat  = out['bhat']
ci_b  = out['CIB']
cib_u = weibull(time1, ci_b[0])
cib_l = weibull(time1, ci_b[1])

# Plots 95% confidence bands
plot(time, growth, 'ko')
plot(time, yhat,   'r-',  lw=2.0, label=r'least squares')
plot(time1, cib_u,  'k--', lw=2.0, label=r'%.f%% CI for $\hat{\beta}$' 
                                         % (100*conf))
plot(time1, cib_l,  'k--', lw=2.0)
xlim([0,16])
xlabel('Time (in days)') 
ylabel('Growth (orgs./ml)') 
title('Growth vs. Time') 
grid()
legend(loc='lower right')
#savefig('../doc/images/prb3e.png', dpi=300)
show()



