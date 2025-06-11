""" Code either inspired by or imported directly from: https://github.com/acids-ircam/variational-timbre.
    Esling, Chemla-Romeu-Santos, Bitton: GENERATIVE TIMBRE SPACES: REGULARIZING VARIATIONAL AUTO-ENCODERS WITH PERCEPTUAL METRICS, 2018"""

from numpy import zeros, where, ndarray
from torch import Tensor
from torch.autograd import Variable

def oneHot(labels, dim):
    if issubclass(type(labels), ndarray):
        n = labels.shape[0]
        t = zeros((n, dim))
        for i in range(n):
            t[i, int(labels[i])] = 1
    elif issubclass(type(labels), Variable):
        n = labels.size(0)
        t = Variable(Tensor(n, dim).zero_())
        for i in range(n):
            t[i, int(labels[i])] = 1
    elif issubclass(type(labels), Tensor):
        n = labels.size(0)
        t = Tensor(n, dim).zero_()
        for i in range(n):
            t[i, int(labels[i])] = 1
    else:
        raise Exception('type %s is not recognized by oneHot function'%type(labels))        
    return t

def fromOneHot(vector):
    if issubclass(type(vector), ndarray):
        ids = where(vector==1)
        return ids[1]
    if issubclass(type(vector), Tensor):
        return vector.eq(1).nonzero()[:, 1]
    return ids[1]