from thinkbayes2.thinkbayes2 import Pmf

pmf = Pmf()
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








