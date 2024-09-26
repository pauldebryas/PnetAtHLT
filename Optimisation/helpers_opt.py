import math

def loss(rate, rate_budget):
    k = math.log(2) / 1.
    if rate <= rate_budget:
        return 0.5-(1./(2*rate_budget))*rate
    if rate > (rate_budget + 1.):
        return 1
    return math.exp(k * (rate - rate_budget)) - 1
