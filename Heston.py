# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 16:12:04 2022

@author: abhishek bansal
"""

import numpy as np

class Heston:
    def __init__(self):
        self.kappa = None
        self.vbar = None
        self.gamma = None
        self.rho = None
        self.v0 = None
        self.S0 = None
        self.rfr = None
        self.maturity = None

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

    def Heston_Char_func(self,u):
        D1 = lambda u: np.sqrt(np.power(self.kappa - self.gamma*self.rho*1j*u,2)
                               + (u**2 + 1j*u)*(self.gamma**2))
        g_minus = lambda u: (self.kappa - 1j*self.rho*self.gamma*u - D1(u))
        g_plus = lambda u: (self.kappa - 1j*self.rho*self.gamma*u + D1(u))
        g = lambda u: g_minus(u)/g_plus(u)
        e_d1 = lambda u: np.exp(-D1(u)*self.maturity)

        val1 = lambda u: ((1j*u*self.rfr*self.maturity) + (self.v0/self.gamma**2)*
                                ((1-e_d1(u))/(1-g(u)*e_d1(u)))*g_minus(u))
        val2 = lambda u: ((self.kappa*self.vbar/self.gamma**2)*
                                (self.maturity*g_minus(u) - 2*np.log((1-g(u)*e_d1(u))/(1-g(u)))))
        cf = np.exp(val1(u))* np.exp(val2(u))
        return cf

    def _adj_Heston_Char_func(self,u):
        return self.Heston_Char_func(u - 1j)/self.Heston_Char_func(-1j)

    def initializeHestonParameters(self, kappa, vbar, gamma, rho,v0,
                                   S0 = None, rfr = None, mat = None):
        self.kappa = kappa
        self._kappa = kappa
        self.vbar = vbar
        self._vbar = vbar
        self.gamma = gamma
        self._gamma = gamma
        self.rho = rho
        self._rho = rho
        self.v0 = v0
        self._v0 = v0
        self.S0 = S0
        self._S0 = S0
        self.rfr = rfr
        self._rfr = rfr
        self.maturity = mat
        self._maturity = mat


