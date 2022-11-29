from numpy import exp, sin
from numpy import linspace, random
from scipy.optimize import leastsq
from lmfit import minimize, Parameters
import matplotlib.pyplot as plt

def residual(variables, x, data, uncertainty):

    """Model a decaying sine wave and subtract data."""
    amp = variables[0]
    phaseshift = variables[1]
    freq = variables[2]
    decay = variables[3]

    model = amp * sin(x*freq + phaseshift) * exp(-x*x*decay)

    return (data-model) / uncertainty

# generate synthetic data with noise
x = linspace(0, 100)
noise = random.normal(size=x.size, scale=0.2)
data = 7.5 * sin(x*0.22 + 2.5) * exp(-x*x*0.01) + noise

# generate experimental uncertainties
uncertainty = abs(0.16 + random.normal(size=x.size, scale=0.05))

variables = [10.0, 0.2, 3.0, 0.007]
out = leastsq(residual, variables, args=(x, data, uncertainty))

params = out[0]
amp = params[0]
phaseshift = params[1]
freq = params[2]
decay = params[3]
fit = amp * sin(x*freq + phaseshift) * exp(-x*x*decay)

plt.plot(fit)
plt.plot(data)
plt.show()