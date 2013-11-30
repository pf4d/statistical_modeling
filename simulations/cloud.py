import sys
src_directory = '../'
sys.path.append(src_directory)

from pylab        import *
from scipy.io     import loadmat
from scipy.stats  import linregress, t
from src.regstats import linRegstats

# ============================================================ #
# Problem 1: Plot of biological recovery percentages vs. time  #
# ============================================================ #
data  = loadmat('../data/cloud.mat')          # Reads in the data
time  = data['cloud']['time'][0][0][0]        # Renames the time variable
time  = time.astype('f')                      # convert to float
recov = data['cloud']['recovery'][0][0][0]    # Renames the recovery variable
fig = figure()
plot(time, recov,'ko')                        # Plots recoveries vs. time
xlim([-5,65])                                 # Sets x-limits on plot
xlabel('Time (in minutes)')
ylabel('Biological Recovery Percentage')
title('Scatterplot of Recovery Percentage vs. Time')
grid()
#savefig('../doc/images/prb1a.png', dpi=300)
show()

fig = figure()
logrecov = log(recov)                         # Log of recoveries
plot(time, logrecov, 'ko')                    # Plots log recovery vs. time
xlim([-5,65])                                 # Sets x-limits on plot
xlabel('Time (in minutes)')
ylabel('Log Biological Recovery Percentage')
title('Scatterplot of Log Recovery vs. Time')
grid()
#savefig('../doc/images/prb1b.png', dpi=300)
show()


# ===================================================================== #
# Problem 1: Regression of log biological recovery percentages vs. time #
# ===================================================================== #
x   = time
y   = logrecov 
out = linRegstats(x, y, 0.95) # get regression statistics for linear model

ci_b = out['CIB']
ci_y = out['CIY']
bhat = out['bhat']
yhat = out['yhat']

# plot the solutions :
plot(x, y, 'ko')
plot(x, yhat, 'r-', lw=2.0, 
     label=r'least squares')
plot(x, ci_b[0,0] + ci_b[0,1]*x, 'k--', lw=2.0, 
     label=r'99% CI for $\hat{\beta}$')
plot(x, ci_b[1,0] + ci_b[1,1]*x, 'k--', lw=2.0)
plot(x, ci_y[0], 'k:', lw=2.0, label=r'99% CI for $\hat{y}$')
plot(x, ci_y[1], 'k:', lw=2.0)
xlim([-5,65])
xlabel('Time (in minutes)')
ylabel('Log Biological Recovery Percentage')
title('Scatterplot of Log Recovery vs. Time')
grid()
legend()
#savefig('../doc/images/prb1e.png', dpi=300)
show()

# compare inside to outside of CI
x_mid         = len(x)/2
min_dyhat_l   = ci_y[0][x_mid]   - yhat[x_mid]
min_dyhat_r   = ci_y[0][x_mid+1] - yhat[x_mid+1]
max_dyhat_l   = ci_y[0][0]  - yhat[0]
max_dyhat_r   = ci_y[0][-1] - yhat[-1]
avg_min_dyhat = (min_dyhat_l + min_dyhat_r) / 2
avg_max_dyhat = (max_dyhat_l + max_dyhat_r) / 2

print avg_min_dyhat / avg_max_dyhat

