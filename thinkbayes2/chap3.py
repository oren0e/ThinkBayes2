from thinkbayes2.thinkbayes2 import Suite

# The Dice Problem #
suite = Dice([4, 6, 8, 12, 20])

class Dice(Suite):
    def Likelihood(self, data, hypo):
        if hypo < data:
            return 0
        else:
            return 1.0/hypo


suite.Update(6)
