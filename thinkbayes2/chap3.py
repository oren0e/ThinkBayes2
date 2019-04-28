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
suite.Print()

for roll in [6, 8, 7, 7, 5, 4]:
    suite.Update(roll)
suite.Print()

# The locomotive problem
hypos = range(1, 1001)


class Train(Suite):
    def Likelihood(self, data, hypo):
        if hypo < data:
            return 0
        else:
            return 1.0/hypo


suite = Train(hypos)
suite.Update(60)
print(suite.Mean())