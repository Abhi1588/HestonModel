# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 16:12:04 2022

@author: abhishek bansal
"""

import numpy as np

"""
Class: Heston Model
Implements Heston model characterstic function, montecarlo methods
:parameters
kappa = mean reversion speed
vbar = long run variance
gamma = vol of vol
rho = correlation
v0 = initial variance
S0 = stock price
rfr = risk free rate
T = time to maturity
"""

def Heston_Char_func(u, kappa, vbar, gamma, rho,v0,rfr,maturity):
    D1 = lambda u: np.sqrt(np.power(kappa - gamma*rho*1j*u,2)
                           + (u**2 + 1j*u)*(gamma**2))
    g_minus = lambda u: (kappa - 1j*rho*gamma*u - D1(u))
    g_plus = lambda u: (kappa - 1j*rho*gamma*u + D1(u))
    g = lambda u: g_minus(u)/g_plus(u)
    e_d1 = lambda u: np.exp(-D1(u)*maturity)

    val1 = lambda u: ((1j*u*rfr*maturity) + (v0/gamma**2)*
                            ((1-e_d1(u))/(1-g(u)*e_d1(u)))*g_minus(u))
    val2 = lambda u: ((kappa*vbar/gamma**2)*
                            (maturity*g_minus(u) - 2*np.log((1-g(u)*e_d1(u))/(1-g(u)))))
    cf = np.exp(val1(u))* np.exp(val2(u))
    return cf

def _adj_Heston_Char_func(self,u, kappa, vbar, gamma, rho,v0,rfr,maturity):
    return Heston_Char_func(u - 1j, kappa, vbar, gamma, rho,v0,rfr,maturity)/Heston_Char_func(-1j, kappa, vbar, gamma, rho,v0,rfr,maturity)

def cir_process(kappa, vbar, gamma, Z, dt, x_i):
    return x_i + kappa*(vbar - x_i)*dt + gamma*np.sqrt(dt)*Z