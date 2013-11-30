from pylab          import *
from scipy.io       import loadmat
from scipy.stats    import t, distributions
from scipy.optimize import leastsq
from scipy.special  import fdtrc, gammaln
from qsturng        import *

chi2cdf = distributions.chi2.cdf

def linRegstats(x, y, conf):
 
  if size(shape(x)) == 1: 
    n    = float(len(x))          # Sample size
  else:
    n    = float(shape(x)[1])     # Sample size
  p      = 2                      # Defines # parameters
  dof    = n - p                  # Degrees of freedom
  nov    = p - 1                  # number of variables
  ym     = mean(y)                # Mean of log recovery
  X      = vstack((ones(n), x)).T # observed x-variable matrix
  bhat   = dot( dot( inv(dot(X.T, X)), X.T), y)
  yhat   = dot(X, bhat)           # Linear fit line
  SSE    = sum((y    - yhat)**2)  # Sum of Squared Errors
  SST    = sum((y    - ym  )**2)  # Sum of Squared Total
  SSR    = sum((yhat - ym  )**2)  # Sum of Squared Residuals (SSR = SST - SSE)
  R2     = SSR / SST              # R^2 Statistic (rval**2)
  MSE    = SSE / dof              # Mean Squared Error (MSE)
  MSR    = SSR / nov              # Mean Squared Residual (MSR)
  F      = MSR / MSE              # F-Statistic
  F_p    = fdtrc(nov, dof, F)     # F-Stat. p-value
  
  # variance of beta estimates :
  VARb   = MSE * inv(dot(X.T, X)) # diag(VARb) == varB
  varB   = diag(VARb)             # variance of beta hat
  seb    = sqrt(varB)             # vector of standard errors for beta hat
 
  # variance of y estimates : 
  VARy   = MSE * dot(dot(X, inv(dot(X.T, X))), X.T)
  varY   = diag(VARy)             # variance of y hat
  sey    = sqrt(varY)             # standard errors for yhat
 
  # calculate t-statistic :
  t_b    = bhat / seb
  t_y    = yhat / sey
 
  # calculate p-values :
  pval   = t.sf(abs(t_b), dof) * 2
  
  tbonf  = t.ppf((1+conf)/2.0, dof)  # uncorrected t*-value
  
  ci_b   = [bhat - tbonf*seb,     # Confidence intervals for betas
            bhat + tbonf*seb]     #   in 2 columns (lower,upper)
  
  ci_y   = [yhat - tbonf*sey,     # Confidence intervals for betas
            yhat + tbonf*sey]     #   in 2 columns (lower,upper)

  resid  = y - yhat

  vara = { 'SSR'   : SSR,
           'SSE'   : SSE,
           'SST'   : SST,
           'df'    : (nov, dof, n-1),
           'MSR'   : MSR,
           'MSE'   : MSE,
           'F'     : F,
           'F_pval': F_p,
           'varY'  : varY,
           'varB'  : varB,
           'SEB'   : seb,
           'SEY'   : sey,
           'tbonf' : tbonf,
           't_beta': t_b,
           't_y'   : t_y,
           'pval'  : pval,
           'CIB'   : array(ci_b),
           'CIY'   : array(ci_y),
           'bhat'  : bhat,
           'yhat'  : yhat,
           'R2'    : R2,
           'resid' : resid}
  return vara


def nonlinRegstats(x, y, f, beta0, conf):

  def residual(beta, x, y, f):
    err = y - f(x, beta)
    return err


  out    = leastsq(residual, beta0, args=(x,y,f), full_output=True)

  bhat   = out[0]
  J      = out[1]
  nfo    = out[2]
  fjac   = nfo['fjac']
  ipvt   = nfo['ipvt']
  msg    = out[3]
  ier    = out[4]
   
  n      = float(len(x))          # Sample size
  p      = float(len(beta0))      # number of parameters
  dof    = max(0, n - p)          # Degrees of freedom
  nov    = p - 1                  # number of variables
  xm     = mean(x)                # Mean of time values
  ym     = mean(y)                # Mean of log recovery
  yhat   = f(x,  bhat)            # non-linear fit line
  SSE    = sum((y    - yhat)**2)  # Sum of Squared Errors
  SST    = sum((y    - ym  )**2)  # Sum of Squared Total
  SSR    = sum((yhat - ym  )**2)  # Sum of Squared Residuals (SSR = SST - SSE)
  R2     = SSR / SST              # R^2 Statistic (rval**2)
  MSE    = SSE / dof              # Mean Squared Error (MSE)
  MSR    = SSR / nov              # Mean Squared Residual (MSR)
  F      = MSR / MSE              # F-Statistic
  F_p    = fdtrc(nov, dof, F)     # F-Stat. p-value
  
  # covariance matrix:
  covB   = MSE * J

  # Vector of standard errors for beta hat (seb) and yhat (sey)
  seb    = sqrt(diag(covB))
  sey    = sqrt(MSE * (1.0/n + (x - xm)**2 / sum((x - xm)**2) ) )
  tbonf  = t.ppf((1+conf)/2.0, dof)  # uncorrected t*-value
  
  # calculate t-statistic :
  t_b    = bhat / seb
  t_y    = yhat / sey
 
  # calculate p-values :
  pval   = t.sf(abs(t_b), dof) * 2
  
  # Confidence intervals
  ci_b   = [bhat - tbonf*seb,
            bhat + tbonf*seb]

  ci_y   = [yhat - tbonf*sey,
            yhat + tbonf*sey]
  
  resid  = y - yhat
  
  vara = { 'SSR'   : SSR,
           'SSE'   : SSE,
           'SST'   : SST,
           'df'    : (nov, dof, n-1),
           'MSR'   : MSR,
           'MSE'   : MSE,
           'F'     : F,
           'F_p'   : F_p,
           'SEB'   : seb,
           'SEY'   : sey,
           't_beta': t_b,
           't_y'   : t_y,
           'pval'  : pval,
           't'     : tbonf,
           'CIB'   : array(ci_b),
           'CIY'   : array(ci_y),
           'bhat'  : bhat,
           'yhat'  : yhat,
           'R2'    : R2,
           'covB'  : covB,
           'resid' : resid}
  return vara


def grpStats(x, y, alpha, interaction=False):

  grp = unique(x)
  sig = std(y)
  t   = float(len(grp))
  na  = float(len(y))
  ymt = mean(y)

  idx   = [] # index list corresponding to group
  yms   = [] # mean of groups
  ss    = [] # standard deviations of groups
  nums  = [] # number of elements in group
  sems  = [] # standard errors of groups
  X     = [] # design matrix
  X.append(ones(na)) # tack on intercept
  for g in grp:
    ii  = where(x == g)[0]
    
    # form a column of the design matrix :
    x_c     = zeros(na)
    x_c[ii] = 1.0
    
    # collect necessary statistics :
    n   = float(len(ii))
    ym  = mean(y[ii])
    s   = std(y[ii])
    sem = s / sqrt(n)
   
    # append to the lists :
    X.append(x_c) 
    idx.append(ii)
    yms.append(ym)
    ss.append(s)
    nums.append(n)
    sems.append(sem)
  
  X    = array(X[:-1]).T         # remove the redundant information 
  idx  = array(idx)
  yms  = array(yms)
  ss   = array(ss)
  nums = array(nums)
  sems = array(sems)

  pair      = []
  pair_mean = []
  hsd_ci    = []

  # sort from largest to smallest
  srt   = argsort(yms)[::-1]
  grp_s = grp[srt]
  yms_s = yms[srt]
  num_s = nums[srt]

  # calculate the Tukey confidence intervals :
  for i,(g1, ym1, n1) in enumerate(zip(grp_s, yms_s, num_s)):
    for g2, ym2, n2 in zip(grp_s[i+1:][::-1], 
                           yms_s[i+1:][::-1], 
                           num_s[i+1:][::-1]):
      p_m = ym1 - ym2
      c   = qsturng(alpha, t, na - t) / sqrt(2) * sig * sqrt(1/n1 + 1/n2)
      pair_mean.append(p_m)
      pair.append(g1 + ' - ' + g2)
      hsd_ci.append(c)
  srt       = argsort(pair_mean)
  pair      = array(pair)[srt]
  pair_mean = array(pair_mean)[srt]
  hsd_ci    = array(hsd_ci)[srt]
  
  # calculate more statistics :
  SSB   = sum( nums * (yms - ymt)**2 )
  SSW   = sum((nums - 1) * ss**2)
  MSW   = SSW / (na - t)
  MSB   = SSB / (t - 1)
  f     = MSB / MSW
  p     = fdtrc((t - 1), (na - t), f) 

  # fit the data to the model :
  muhat = dot( dot( inv(dot(X.T, X)), X.T), y)
  yhat  = dot(X, muhat)
  resid = y - yhat
  
  vara = {'grp_means'  : yms,
          'grp_SDs'    : ss,
          'grp_SEMs'   : sems,
          'grp_names'  : grp,
          'grp_lens'   : nums,
          'grp_dof'    : t - 1,
          'dof'        : na - t,
          'F'          : f,
          'MSW'        : MSW,
          'MSB'        : MSB,
          'alpha'      : alpha,
          'pval'       : p,
          'pairs'      : pair,
          'pair_means' : pair_mean,
          'HSD_CIs'    : hsd_ci,
          'muhat'      : muhat,
          'yhat'       : yhat,
          'resid'      : resid}
  return vara

def anovan(x, y, factor_names, conf, interaction=False):
  
  ym  = mean(y)
  SST = sum((y - ym)**2)

  # find the indexes to each of the groups within each treatment :
  # n-way analysis 
  if type(x) == list:
    tmt_names = []
    tmt_idxs  = []
    tmt_means = []
    tmt_lens  = []
    X         = []                     # design matrix
    na        = float(shape(x)[1])     # Sample size
    X.append(ones(na))                 # tack on intercept
    for x_i in x:
      types = unique(x_i)
      tmt_names.append(types)
      ii    = []
      means = []
      lens  = []
      for t in types:
        i       = where(x_i == t)[0]
        x_c     = zeros(na)
        x_c[i]  = 1.0
        ii.append(i)
        lens.append(len(i))
        means.append(mean(y[i]))
        X.append(x_c)
      X = X[:-1]         # remove the redundant information 
      tmt_idxs.append(array(ii))
      tmt_means.append(array(means))
      tmt_lens.append(array(lens))
    tmt_names = array(tmt_names)
    tmt_idxs  = array(tmt_idxs)
    tmt_means = array(tmt_means)
    tmt_lens  = array(tmt_lens)
    
    # sum of squares between cells :
    SSB = 0
    a   = len(tmt_idxs[0])
    b   = len(tmt_idxs[1])
    dfT = len(y) - 1
    dfA = a - 1
    dfB = b - 1
    dfAB = dfA * dfB
    dfE  = len(y) - a * b
    cell_means = []
    for l1 in tmt_idxs[0]:
      c_m = []
      for l2 in tmt_idxs[1]:
        ii = intersect1d(l1, l2)
        if ii.size != 0:
          c_m.append(mean(y[ii]))
          SSB += len(y[ii]) * (mean(y[ii]) - ym)**2
      cell_means.append(array(c_m))
    cell_means = array(cell_means)


  # one-way analysis
  else:
    na        = float(len(x))
    tmt_names = unique(x)
    X         = []     # design matrix
    X.append(ones(na)) # tack on intercept
    for t in tmt_names:
      ii  = where(x == t)[0]
      
      # form a column of the design matrix :
      x_c     = zeros(na)
      x_c[ii] = 1.0
      
      # append to the lists :
      X.append(x_c) 
    X = X[:-1]         # ensure non-singular matrix 

  # add rows for interaction terms :
  if interaction and type(x) == list:
    k = 0
    for t in tmt_names[:-1]:
      k += len(t)
      for i, x1 in enumerate(X[1:k]):
        for x2 in X[k:]:
          X.append(x1 * x2)
  X = array(X).T       # design matrix is done

  # calculate statistics :
  SS = array([])
  inter_names = []
  for il, nl, name, mul in zip(tmt_idxs, tmt_lens, factor_names, tmt_means):
    SS = append(SS, sum( nl*(mul - ym)**2))
    inter_names.append(name)
  if interaction:
    inter_names.append(inter_names[0] + ' x ' + inter_names[1])
    SS = append(SS, SSB - SS[0] - SS[1])
  
  # fit the data to the model :
  muhat = dot( dot( inv(dot(X.T, X)), X.T), y)
  yhat  = dot(X, muhat)
  resid = y - yhat
  SSE   = SST - sum(SS)

  # calculate mean-squares :
  MSA  = SS[0] / dfA
  MSB  = SS[1] / dfB
  MSE  = SSE   / dfE
 
  # calculate F-statistics :
  FA  = MSA  / MSE
  FB  = MSB  / MSE

  # calculate p-values:
  pA   = fdtrc(dfA, dfE, FA)
  pB   = fdtrc(dfB, dfE, FB)
  
  if interaction :
    MSAB = SS[2] / dfAB
    FAB  = MSAB / MSE
    pAB  = fdtrc(dfAB, dfE, FAB)
    vara = {'tmt_names' : tmt_names,
            'tmt_means' : tmt_means,
            'tmt_lens'  : tmt_lens,
            'tmt_idxs'  : tmt_idxs,
            'cell_means': cell_means,
            'SST'       : SST,
            'SSB'       : SSB,
            'SSE'       : SSE,
            'SS'        : SS,
            'MSA'       : MSA,
            'MSB'       : MSB,
            'MSAB'      : MSAB,
            'MSE'       : MSE,
            'FA'        : FA,
            'FB'        : FB,
            'FAB'       : FAB,
            'dfA'       : dfA,
            'dfB'       : dfB,
            'dfAB'      : dfAB,
            'dfE'       : dfE,
            'dfT'       : dfT,
            'pA'        : pA,
            'pB'        : pB,
            'pAB'       : pAB,
            'i_names'   : inter_names,
            'muhat'     : muhat,
            'yhat'      : yhat,
            'resid'     : resid}
  else :  
    vara = {'tmt_names' : tmt_names,
            'tmt_means' : tmt_means,
            'tmt_lens'  : tmt_lens,
            'tmt_idxs'  : tmt_idxs,
            'cell_means': cell_means,
            'SST'       : SST,
            'SSB'       : SSB,
            'SSE'       : SSE,
            'SS'        : SS,
            'MSA'       : MSA,
            'MSB'       : MSB,
            'MSE'       : MSE,
            'FA'        : FA,
            'FB'        : FB,
            'dfA'       : dfA,
            'dfB'       : dfB,
            'dfE'       : dfE,
            'dfT'       : dfT,
            'pA'        : pA,
            'pB'        : pB,
            'i_names'   : inter_names,
            'muhat'     : muhat,
            'yhat'      : yhat,
            'resid'     : resid}
  return vara


# ===================================== #
# Negative Binomial Likelihood Function #
# ===================================== #
def negbinlike(th, y):
  """
  Finds the likelihood for a given vector <th> of parameter values.
  So, <th> is nothing more than (r,p) in the negative binomial.
  """
  n     = len(y)
  r     = th[0]
  p     = th[1]
  nt1   = n*r*log(p) + sum(y)*log(1-p) - n*gammaln(r)
  nt2   = sum(gammaln(r + y))
  nlike = -(nt1 + nt2)
  return nlike

def chiTest(data, dist, max_val, num_param):
  """
  perform chi^2 test for distribution <dist> on data <data>.  <max_val> is the
  maximum value with frequency greater than 5 and <num_param> is the number
  of parameters for distribution.  Returns probability the  distribution is not   distributed like <dist>.
  """
  px       = append(dist, 1-sum(dist))          # add on results >= 11
  expfreq  = px * len(data)                     # expected frequency
  
  red_nums = range(max_val + 1)
  red_nums.append(max(data))
  
  obsfreq, n, mid = hist(data, red_nums)              # histogram
  D               = sum((obsfreq-expfreq)**2/expfreq) # chi squared
  m               = len(n) - 1                        # number of outcomes
  dof             = m - (num_param + 1)               # degrees of freedom
  pval            = 1 - chi2cdf(D, dof)               # probability not Poissson
  return pval

def plot_means(ax, mu, c, labels, label_size, tit, xlab):
  
  for i, (m, ci) in enumerate(zip(mu, c)):
    j = i+1
    cu = m + ci
    cl = m - ci
    ax.plot([cl, cu], [j, j], 'k-', lw=2.0)
    ax.plot(m,        j,      'ro')
 
  ax.set_ylim([0,len(mu)+1]) 
  ax.set_yticks(range(1,len(mu)+1))
  ax.set_yticklabels(labels)
  for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(label_size)
  ax.set_title(tit)
  ax.set_xlabel(xlab)
  grid()
  tight_layout()


def interactionPlot(ax, means, names, xlab):
  l = len(means)
  for m,n in zip(means.T, names[1]):
    plot(range(1, l+1), m, lw=2.0, label=n)
  ax.set_xlim([.85, l+.15]) 
  ax.set_xticks(range(1, l+1))
  ax.set_xticklabels(names[0])
  for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(15)
  ax.set_title('Interaction Plot')
  ax.set_xlabel(xlab)
  ax.set_ylabel('mean')
  leg = legend()
  leg.get_frame().set_alpha(0.5)
  tight_layout()


class prbplotObj:
  """
  class wraps the methods of an AxesSubplot to those of pyplot.
  """
  def __init__(self, ax):
    self.ax = ax
  
  def plot(self, x1, y1, t, x2, y2):
    self.ax.plot(x1, y1, t, x2, y2)
  
  def title(self, tit):
    self.ax.set_title(tit)

  def xlabel(self, xlab):
    self.ax.set_xlabel(xlab)

  def ylabel(self, ylab):
    self.ax.set_ylabel(ylab)

  def text(self, x, y, txt):
    self.ax.text(x, y, txt)


