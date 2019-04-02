from thinkbayes2.thinkbayes2 import Pmf

pmf = Pmf('ab')
for x in [1,2,3,4,5,6]:
    pmf.Set(x, 1/6.0)

pmf = Pmf()
word_list = ['this', 'is', 'me']
for word in word_list:
    pmf.Incr(word, 1)
pmf.Normalize()
print(pmf.Prob('this'))

# The cookie problem
pmf = Pmf()
pmf.Set('Bowl 1', 0.5)
pmf.Set('Bowl 2', 0.5)

pmf.Mult('Bowl 1', 0.75)
pmf.Mult('Bowl 2', 0.5)
pmf.Normalize()
print(pmf.Prob('Bowl 1'))

class Cookie(Pmf):

    def __init__(self, hypos):
        Pmf.__init__(self, hypos)
        for hypo in hypos:
            self.Set(hypo, 1)
        self.Normalize()

    def Update(self, data):
        for hypo in self.Values():
            like = self.Likelihood(data, hypo)
            self.Mult(hypo, like)
        self.Normalize()

    mixes = {
        'Bowl 1': dict(vanilla=0.75, chocolate=0.25),
        'Bowl 2': dict(vanilla=0.5, chocolate=0.5),
    }

    def Likelihood(self, data, hypo):
        mix = self.mixes[hypo]
        like = mix[data]
        return like

pmf = Cookie(['Bowl 1','Bowl 2'])
pmf.Update('vanilla')
pmf.Print()

for hypo, prob in pmf.Items():
    print(hypo, prob)

dataset = ['vanilla','chocolate', 'vanilla']
for data in dataset:
    pmf.Update(data)


# Monty Hall Problem #
class Monty(Pmf):
    def __init__(self, hypos):
        Pmf.__init__(self)
        for hypo in hypos:
            self.Set(hypo, 1)
        self.Normalize()

    def Update(self, data):
        for hypo in self.Values():
            like = self.Likelihood(data, hypo)
            self.Mult(hypo, like)
        self.Normalize()

    def Likelihood(self, data, hypo):
        if hypo == data:
            return 0
        elif hypo == 'A':
            return 0.5
        else:
            return 1


hypos = 'ABC'
pmf = Monty(hypos)
data = 'B'  # Monty opens door B
pmf.Update(data)

# Print the result
for hypo, prob in pmf.Items():
    print(hypo, prob)

# Because several parts are the same we can encapsulate them in an object:
class Suite(Pmf):
    """Represents a suite of hypotheses and their probabilities."""

    def __init__(self, hypos = tuple()):
        """Initializes the distribution."""
        Pmf.__init__(self)
        for hypo in hypos:
            self.Set(hypo, 1)
        self.Normalize()

    def Update(self, data):
        """Updates each hypothesis based on the data."""
        for hypo in self.Values():
            like = self.Likelihood(data, hypo)
            self.Mult(hypo, like)
        self.Normalize()

    def Print(self):
        """Prints the hypotheses and their probabilities."""
        for hypo, prob in sorted(self.Items()):
            print(hypo, prob)

# To use Suite we should write a class that inherits from it and provides Likelihood method.
# Here is a solution for the Monty-Hall problem:

from thinkbayes2.thinkbayes2 import Suite


class Monty(Suite):
    def Likelihood(self, data, hypo):
        if hypo == data:
            return 0
        elif hypo == 'A':
            return 0.5
        else:
            return 1


suite = Monty('ABC')
suite.Update('B')
suite.Print()

# The M&M problem
class M_and_M(Suite):
    mix94 = dict(brown=30, yellow=20, red=20, green=10, orange=10, tan=10)
    mix96 = dict(blue=24, green=20, orange=16, yellow=14, red=13, brown=13)

    hypoA = dict(bag1=mix94, bag2=mix96)
    hypoB = dict(bag1=mix96, bag2=mix94)

    hypotheses = dict(A=hypoA, B=hypoB)

    def Likelihood(self, data, hypo):
        bag, color = data
        mix = self.hypotheses[hypo][bag]
        like = mix[color]
        return like


suite = M_and_M('AB')
suite.Update(('bag1', 'yellow'))
suite.Update(('bag2', 'green'))
suite.Print()