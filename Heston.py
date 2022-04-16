# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 16:12:04 2022

@author: abhishek bansal
"""

import numpy as np

class Heston:
    def __init__(self):
        self.kappa = None
        self.theta = None
        self.sigma = None
        self.rho = None
        self.v0 = None
        self.S0 = None
        self.rfr = None
        self.maturity = None

        """
        Class: Heston Model
        Implements Heston model characterstic function, montecarlo methods
        :parameters
        kappa = 
        theta = 
        sigma = 
        rho = 
        v0 = 
        S0 = 
        rfr = 
        T = 
        """

    def _a(self):
        try:
            return (self.sigma**2)/2
        except:
            print("Heston parameters are not defined")

    def _b(self,u):
        try:
            return self.kappa - self.rho*self.sigma*1j*u
        except:
            print("Error _b")

    def _c(self,u):
        try:
            return -(u**2+1j*u)/2
        except:
            print("error _c")

    def _d(self,u):
        try:
            return np.sqrt(self._b(u)**2 - 4*self._a()*self._c(u))
        except:
            print("Error _d")

    def _xminus(self,u):
        try:
            return (self._b(u) - self._d(u))/(2*self._a())
        except:
            print("Error xminus")

    def _xplus(self,u):
        try:
            return (self._b(u) + self._d(u))/(2*self._a())
        except:
            print("Error xplus")

    def _g(self,u):
        try:
            return self._xminus(u)/self._xplus(u)
        except:
            print("Error _g")

    def _C(self,u):
        try:
            val = self.maturity*self._xminus(u) - np.log((1-self._g(u)*np.exp(-self.maturity*self._d(u)))
                                                  /(1-self._g(u)))/self._a()
            return self.rfr*self.maturity*1j*u + self.theta*self.kappa*val
        except:
            print("Error _C")

    def _D(self,u):
        try:
            val1 = 1 - np.exp(-self.maturity*self._d(u))
            val2 = 1 - self._g(u)*np.exp(-self.maturity*self._d(u))
            return (val1/val2)*self._xminus(u)
        except:
            print("Error _D")
    def Heston_Char_func(self,u):
        return np.exp(self._C(u) + self._D(u)*self.v0 + 1j*np.log(self.S0))

    def _adj_Heston_Char_func(self,u):
        return self.Heston_Char_func(u - 1j)/self.Heston_Char_func(-1j)

    def initializeHestonParameters(self, kappa, theta, sigma, rho,v0,
                                   S0 = None, rfr = None, mat = None):
        self.kappa = kappa
        self._kappa = kappa
        self.theta = theta
        self._theta = theta
        self.sigma = sigma
        self._sigma = sigma
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


