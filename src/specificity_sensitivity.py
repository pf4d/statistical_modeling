from pylab import *

def get_results(A,B,C,D):
  """
  INPUTS:
    A : number of true-positives
    B : number of false-positives
    C : number of false-negatives
    D : number of true-negative
  OUTPUT:
    tuple (sensitivity, specificity, observed_prevalance, true_prevalence
  """
  N         = A + B + C + D
  sens      = A/(A+C)
  spec      = D/(B+D)
  ob_prev   = (A+B)/N
  true_prev = (-1 + (A+B)/N + D/(B+D)) / (A/(A+C) - B/(B+D))
  return sens, spec, ob_prev, true_prev

