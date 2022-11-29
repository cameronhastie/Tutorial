from numpy import exp, sin
from numpy import linspace, random
from lmfit import minimize, Parameters
import matplotlib.pyplot as plt

def residual(params, x, data, uncertainty):
    amp = params['amp']
    phaseshift = params['phase']
    freq = params['frequency']
    decay = params['decay']

    model = amp * sin(x*freq + phaseshift) * exp(-x*x*decay)
 
    return (data-model) / uncertainty

# generate synthetic data with noise
x = linspace(0, 100)
noise = random.normal(size=x.size, scale=0.2)
data = 7.5 * sin(x*0.22 + 2.5) * exp(-x*x*0.01) + noise

# generate experimental uncertainties
uncertainty = abs(0.16 + random.normal(size=x.size, scale=0.05))

variables = [10.0, 0.2, 3.0, 0.007]

params = Parameters()
params.add('amp', value=10)
params.add('decay', value=0.007)
params.add('phase', value=0.2)
params.add('frequency', value=3.0)

out = minimize(residual, params, args=(x, data, uncertainty))

fit_params = out.params
amp = fit_params['amp']
phaseshift = fit_params['phase']
freq = fit_params['frequency']
decay = fit_params['decay']
fit = amp * sin(x*freq + phaseshift) * exp(-x*x*decay)

plt.plot(fit)
plt.plot(data)
plt.show()