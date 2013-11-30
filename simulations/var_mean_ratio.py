from pylab       import *
from scipy.stats import scoreatpercentile

# save the data for later analysis :
ratios = []
mrat   = []
sdrat  = []
low    = []
upp    = []

# create figure and subplots :
fig = figure(figsize=(12,8))
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(235)

theta = 20                             # Assigns the Poisson mean at 20
axs   = [ax1, ax2, ax3, ax4, ax5]      # axes for plotting
ns    = [10, 25, 50, 100, 500]         # Vector of 5 sample sizes
iss   = range(1,6)                     # indexes for plotting only

for i, n, ax in zip(iss, ns, axs):     # Begins loop through n-values
  nsim  = 1000                         # Sets the number of simulations
  ratio = []                           # initialize ratio array
                                            
  for j in range(nsim):                       # Loops through 1000 simulations
    dat = poisson(theta, n)                   # Generates n Poisson(20) values
    ratio.append(var(dat) / mean(dat))        # Computes var-to-mean ratio
                                              
  mrat.append(mean(ratio))                    # Computes mean of 1000 ratios
  sdrat.append(std(ratio))                    # Computes SD of 1000 ratios
  ratio = sort(ratio)                         # Sorts the 1000 ratios
  ratios.append(ratio)                        # append the ratio to main array
  low.append(ratio[round(nsim * 0.025)])      # Ratio CI lower limit
  upp.append(ratio[round(nsim * 0.975)])      # Ratio CI upper limit

  ax.hist(ratio, 10)                          # Histogram of the ratios
  if i != 1 and i != 2:
    ax.set_xlabel('Variance-to-Mean Ratios')  # x-axis label on plot
  if i == 1 or i == 4:
    ax.set_ylabel('Frequency')                # y-axis label on plot
  ax.set_title(r'Poisson Ratios ($n= ' + str(n) + '$)')
  ax.set_xlim([0, 2.5])                       # Sets x-axis limits
  ax.set_ylim([0, 300])                       # Sets y-axis limits
  ax.grid()                                   # add a grid

ci = array(upp) - array(low)                  # calculate the confidence int.
#savefig('../doc/images/prb4.png', dpi=300)
show()
