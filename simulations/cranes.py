import sys
src_directory = '../'
sys.path.append(src_directory)

from pylab        import *
from scipy.io     import loadmat
from scipy.stats  import linregress, t, probplot
from src.regstats import *

# ============================================================ #
# Problem 2: Plot of crane populations vs. time                #
# ============================================================ #
data   = loadmat('../data/cranes.mat')['cranes'][0][0]
cranes = data[1].flatten()                    # number of cranes
time   = data[0].flatten()                    # convert to float
logcranes = log(cranes)                       # Log of cranes

x   = time - min(time)
y   = cranes 
out = linRegstats(x, y, 0.95) # get regression statistics for linear model

fig = figure()
plot(time, cranes,      'ko',         label='data')
plot(time, out['yhat'], 'r-', lw=2.0, label='best-fit')
xlabel('year')
ylabel('Crane Population')
title('Scatterplot of Cranes vs. Time')
leg = legend(loc='upper left')
leg.get_frame().set_alpha(0.5)
grid()
tight_layout()
#savefig('../doc/images/prb2b.png', dpi=300)
show()

y   = logcranes 
out = linRegstats(x, y, 0.95) # get regression statistics for linear model

fig = figure()
plot(time, logcranes,   'ko',         label='data')
plot(time, out['yhat'], 'r-', lw=2.0, label='best-fit')
xlabel('year')
ylabel('Log Crane Population')
title('Scatterplot of Log Cranes vs. Time')
leg = legend(loc='upper left')
leg.get_frame().set_alpha(0.5)
grid()
tight_layout()
#savefig('../doc/images/prb2c.png', dpi=300)
show()


# =========================================================== #
# Problem 2f - Residual and normal quantile plot of residuals #
# =========================================================== #
fig = figure(figsize=(12,5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(out['yhat'], out['resid'], 'ko')
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
tight_layout()
#savefig('../doc/images/prb2f.png', dpi=300)
show()


# ===================================================================== #
# Problem 2g: Regression of log crane population vs. time                #
# ===================================================================== #
ci_y = out['CIY']
yhat = out['yhat']

# plot the solutions :
plot(time, y,       'ko')
plot(time, yhat,    'r-', lw=2.0, label=r'least squares')
plot(time, ci_y[0], 'k:', lw=2.0, label=r'95% CI for $\hat{y}$')
plot(time, ci_y[1], 'k:', lw=2.0)
xlabel('year')
ylabel('Log Crane Population')
title('Scatterplot of Log Cranes vs. Time')
leg = legend(loc='upper left')
leg.get_frame().set_alpha(0.5)
grid()
tight_layout()
#savefig('../doc/images/prb2g.png', dpi=300)
show()


# ======================================================================= #
# Problem 2h,i - Individual 95% Confidence intervals for the 3 parameters #
# ======================================================================= #

def exp_growth(t, beta):
  a = beta[0]
  b = beta[1]
  g = beta[2]
  y = a +  b*t**g
  return y

x   = time - min(time)
y   = cranes 

beta0 = [18.0, 0.0, 1.0]                # Parameter starting values
conf  = 0.95                            # confidence region

# Performs nonlinear fit returning betahats, resids
out = nonlinRegstats(x, y, exp_growth, beta0, conf)

# Computes predicted values (yhat's)
yhat  = out['yhat']
ci_y  = out['CIY']
ciy_u = out['CIY'][0]
ciy_l = out['CIY'][1]

# Plots 95% confidence bands
plot(time, y,     'ko')
plot(time, yhat,  'r-',  lw=2.0, label=r'least squares')
plot(time, ciy_u, 'k--', lw=2.0, label=r'%.f%% CI for $\hat{y}$' % (100*conf))
plot(time, ciy_l, 'k--', lw=2.0)
xlabel('year')
ylabel('Crane Population')
title('Scatterplot of Cranes vs. Time')
leg = legend(loc='upper left')
leg.get_frame().set_alpha(0.5)
grid()
tight_layout()
#savefig('../doc/images/prb2i.png', dpi=300)
show()


# =========================================================== #
# Problem 2j - Residual and normal quantile plot of residuals #
# =========================================================== #
fig = figure(figsize=(12,5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(out['yhat'], out['resid'], 'ko')
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
tight_layout()
#savefig('../doc/images/prb2j.png', dpi=300)
show()

