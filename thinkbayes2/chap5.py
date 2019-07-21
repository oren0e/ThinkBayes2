# Addends

"""
Simulation: Given a Pmf that represents the distribution for a single die, you can draw random samples,
 add them up, and accumulate the distribution of simulated sums.
"""
import thinkbayes2

class Die(thinkbayes2.Pmf):
    def __init__(self, sides):
        thinkbayes2.Pmf.__init__(self)
        for x in range(1, sides+1):
            self.Set(x, 1)
        self.Normalize()


d6 = Die(6)
dice = [d6] * 3
three = thinkbayes2.SampleSum(dice, 1000)

"""
The other approach is Enumeration. Enumerate all pairs of values and compute the sum and probability of each pair.
This is implemented in Pmf.__add__
"""
three_exact = d6 + d6 + d6
# -------------------------------
# plot the two distributions
import matplotlib.pyplot as plt
import numpy as np

keys = three_exact.GetDict().keys()
vals = three_exact.GetDict().values()
keys_sample = three.GetDict().keys()
vals_sample = three.GetDict().values()

plt.plot(keys,vals,'g--', linewidth=2)
plt.plot(keys_sample,vals_sample[np.argsort(list(keys_sample))],'r',linewidth=2) # does not work
plt.show()
# ------------------------------

# Maxima

# The code to simulate the maxima is almost identical to the code of simulating sums
best_attr_cdf = three_exact.Max(6)
best_attr_pmf = best_attr_cdf.MakePmf()

keys = best_attr_pmf.GetDict().keys()
vals = best_attr_pmf.GetDict().values()

plt.plot(keys,vals,'b', linewidth=3)
plt.xlabel('Sum of three d6', fontsize=18)
plt.ylabel('Probability', fontsize=16)
plt.show()

# Mixtures
d6 = Die(6)
d8 = Die(8)

# create a Pmf to represent the mixture
mix = thinkbayes2.Pmf()
for die in [d6, d8]:
    for outcome, prob in die.Items():
        mix.Incr(outcome, prob)
mix.Normalize()

# we create a Pmf that maps from each die to the probability it is selected
pmf_dice = thinkbayes2.Pmf()
pmf_dice.Set(Die(4), 5)
pmf_dice.Set(Die(6), 4)
pmf_dice.Set(Die(8), 3)
pmf_dice.Set(Die(12), 2)
pmf_dice.Set(Die(20), 1)
pmf_dice.Normalize()

# next, we need a more general version of the mixture algorithm
mix = thinkbayes2.Pmf()
for die, weight in pmf_dice.Items():
    for outcome, prob in die.Items():
        mix.Incr(outcome, weight*prob)

