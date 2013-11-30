import sys
src_directory = '../'
sys.path.append(src_directory)

from pylab          import *
from scipy.optimize import leastsq
from scipy.io       import loadmat
from src.regstats   import *
from scipy.stats    import probplot


# ============================================= #
# Problem 2: Tuberculosis fitness cost analysis #
# ============================================= #
data       = loadmat('../data/gagneux.mat')
gag        = data['gag'][0][0]
strain     = gag[0]
mutation   = gag[1]
fit        = gag[2].flatten()
background = gag[3]

mut  = []
bkg  = []
alg  = []
strn = []
for m, b, s in zip(mutation, background, strain):
  mut.append(str(m[0][0]))
  bkg.append(str(b[0][0]))
  strn.append(str(s[0][0]))
  alg.append(str(m[0][0] + '-' + b[0][0]))
alg  = array(alg)
mut  = array(mut)
bkg  = array(bkg)
strn = array(strn)


# ================================================================== #
# a: Exploratory analysis comparison of relative fitness by mutation #
#    and background (all 13 combinations), including a plot of the   #
#    group means, and table of means and standard deviations.        #
# ================================================================== #
out = grpStats(alg, fit, 0.95)

# plot the group statistics :
grp_means = out['grp_means']
grp_names = out['grp_names']
grp_SDs   = out['grp_SDs']

fig = figure(figsize=(6,8))
ax  = fig.add_subplot(111)

tit      = 'EDA - Group Means and one S.D.'

plot_means(ax, grp_means, grp_SDs, grp_names, 12, tit, 'means')
#savefig('../doc/images/prb2a1.png', dpi=300)
show()

# plot the joint-group CIs
pairs      = out['pairs']
pair_means = out['pair_means']
cis        = out['HSD_CIs']
alpha      = out['alpha']

fig = figure(figsize=(6,8))
ax  = fig.add_subplot(111)

tit = '%.f%% HSD Joint Confidence Intervals' % (100*alpha)
 
plot_means(ax, pair_means, cis, pairs, 5, tit, r'$\Delta$ means')
#savefig('../doc/images/prb2a2.png', dpi=300)
show()


# ==================================================================== #
# b: Exploratory analysis comparison of relative fitness by mutation   #
#      and a 1-way ANOVA of relative fitness on mutation group to test #
#      for differences in the 9 mutations                              #
# ==================================================================== #
fit1 = fit[where(bkg == 'CDC')]
mut1 = mut[where(bkg == 'CDC')]

out = grpStats(mut1, fit1, 0.95)

pairs      = out['pairs']
pair_means = out['pair_means']
cis        = out['HSD_CIs']
alpha      = out['alpha']

fig = figure(figsize=(6,8))
ax  = fig.add_subplot(111)

tit = '%.f%% HSD Joint Confidence Intervals' % (100*alpha)
 
plot_means(ax, pair_means, cis, pairs, 10, tit, r'$\Delta$ means')
#savefig('../doc/images/prb2b.png', dpi=300)
show()


# =================================================================== #
# c: Residual plot and normal quantile plot for 1-way ANOVA residuals #
# =================================================================== #
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
#savefig('../doc/images/prb2c.png', dpi=300)
show()


# =================================================================== #
# d: Reruns the 1-way ANOVA model above without the R529Q mutation,   #
#    giving the ANOVA table, a figure of multiple comparisons between #
#    fitness means, a residual plot, and a normal quantile plot.      #
# =================================================================== #
ii   = where(mut != 'R529Q')[0]
fit2 = fit[ii]
mut2 = mut[ii]

out2 = grpStats(mut2, fit2, 0.95)

pairs      = out2['pairs']
pair_means = out2['pair_means']
cis        = out2['HSD_CIs']
alpha      = out2['alpha']

fig = figure(figsize=(6,8))
ax  = fig.add_subplot(111)

tit = '%.f%% HSD Joint Confidence Intervals' % (100*alpha)
 
plot_means(ax, pair_means, cis, pairs, 10, tit, r'$\Delta$ means')
#savefig('../doc/images/prb2d1.png', dpi=300)
show()

fig = figure(figsize=(12,5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(out2['yhat'], out2['resid'], 'ko')
ax1.set_xlabel('Predicted Values') 
ax1.set_ylabel('Residuals') 
ax1.set_title('Residual Plot') 
ax1.grid()

# Normal quantile plot of residuals
p = prbplotObj(ax2)
probplot(out2['resid'], plot=p)
ax2.set_xlabel('Standard Normal Quantiles') 
ax2.set_ylabel('Residuals') 
ax2.set_title('Normal Quantile Plot')
ax2.grid()
#savefig('../doc/images/prb2d2.png', dpi=300)
show()


# ======================================================================== #
# e: Prepares data for 2-way ANOVA on mutation and background, keeping     #
#    only the 4 mutations common to both backgrounds, namely S531L, H526Y, #
#    H526D, and S531W, and runs a 2-way ANOVA with an interaction.         #
# ======================================================================== #
i1 = where(mutation == 'S531L')[0]
i2 = where(mutation == 'H526Y')[0]
i3 = where(mutation == 'H526D')[0]
i4 = where(mutation == 'S531W')[0]
ii = hstack((i1,i2,i3,i4))

out = anovan([bkg[ii], mut[ii]], fit[ii], 
             ['background', 'mutation'], 0.95, True)

c_m = out['cell_means']
c_n = out['tmt_names']

fig = figure()
ax  = fig.add_subplot(111)

interactionPlot(ax, c_m, c_n, 'Background')
#savefig('../doc/images/prb2e.png', dpi=300)
show()

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
#savefig('../doc/images/prb2e.png', dpi=300)
show()

#obsnum = find(strcmp(gag.mutation,'S531L')+...
#              strcmp(gag.mutation,'H526Y')+...
#              strcmp(gag.mutation,'H526D')+...
#              strcmp(gag.mutation,'S531W'));
#mutation = gag.mutation(obsnum);
#fitness = gag.fitness(obsnum);
#background = gag.background(obsnum);
#[p3,tab3,stats3,terms3] = anovan(fitness,{mutation,background}, ...
#    'varnames',{'mutation','background'}, ...
#    'model','interaction');
#
## ============================================================ #
## f: Constructs an interaction plot to examine the interaction #
##    between mutation and background on relative fitness.      #
## ============================================================ #
#mut2 = [ones(6,1);2*ones(7,1);3*ones(6,1);4*ones(6,1);...
#        ones(5,1);2*ones(4,1);3*ones(4,1);4*ones(4,1)];
#back2 = [ones(25,1);2*ones(17,1)];
#interactionplot(fitness,{mutation background}, ...
#    'varnames',{'mutation','background'});
