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

