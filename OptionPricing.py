# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 20:12:04 2022

@author: abhishek bansal
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from Heston import Heston
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

outputPath = r"output/"

class EuroOption(Heston):
    def __init__(self, stockPrice, strike, vol, rate, maturity, dividend=0,optionType = "call"):
        self.S0 = stockPrice
        self.strike = strike
        self.rfr = rate
        self.vol = vol
        self.maturity = maturity
        self.q = dividend
        if optionType == "call":
            self.optionType = 1
        elif optionType == "put":
            self.optionType = -1

    def price_BS_analytical(self,S0=None,strike=None,rfr=None,vol=None,
                            maturity=None,optionType=None):
        S0 = self.S0 if S0 is None else S0
        strike = self.strike if strike is None else strike
        rfr = self.rfr if rfr is None else rfr
        vol = self.vol if vol is None else vol
        maturity = self.maturity if maturity is None else maturity
        optionType = self.optionType if optionType is None else optionType

        d1 = ((np.log(S0/strike)+(rfr + (vol**2)/2)*maturity)
              /(vol*(np.sqrt(maturity))))
        d2 = d1 - vol*np.sqrt(maturity)

        return optionType*(S0*norm.cdf(optionType*d1)
                                - strike*np.exp(-rfr*maturity)*norm.cdf(optionType*d2))

    def BS_vega(self,S0=None,strike=None,rfr=None,vol=None,
                            maturity=None,optionType=None):

        S0 = self.S0 if S0 is None else S0
        strike = self.strike if strike is None else strike
        rfr = self.rfr if rfr is None else rfr
        vol = self.vol if vol is None else vol
        maturity = self.maturity if maturity is None else maturity
        optionType = self.optionType if optionType is None else optionType

        d1 = ((np.log(S0/strike)+(rfr + (vol**2)/2)*maturity)
              /(vol*(np.sqrt(maturity))))
        return (S0*norm.pdf(d1)*np.sqrt(maturity))[:,0]

    def _getBS_ImpVol(self,targetPrice,MAX_ITERATIONS = 200,PRECISION = 1.0e-12):
        #sigma = np.empty_like(self.price_BS_analytical().shape)
        initGuess= 0.5
        sigma = np.full_like(self.price_BS_analytical(), initGuess)# empty_like(self.price_BS_analytical().shape)
        def objFunc(sigma):
            sigma = sigma.reshape([len(sigma),1])
            price = self.price_BS_analytical(vol=sigma)
            diff = price - targetPrice.reshape(price.shape)
            return sum(abs(diff))
        #print(objFunc(sigma).shape)
        options = {'maxiter': MAX_ITERATIONS}
        res = minimize(objFunc, sigma, method='BFGS', options=options,tol=PRECISION)
        # for i in range(0, MAX_ITERATIONS):
        #     price = self.price_BS_analytical(vol=sigma)[:,0]
        #     vega = self.BS_vega(vol=sigma)
        #     diff = price - targetPrice  # root
        #     print((diff**2 < PRECISION).all())
        #     if (diff**2 < PRECISION).all():
        #         print("Price Diff is {}".format(diff))
        #         return sigma
        #     sigma = sigma - diff / vega  # f(x) / f'(x)
        return res.x

    def getBS_IV(self):


        return


    def price_Heston_CosMethod(self,a,b,N):

        def upsilon_n(a,b,c,d,k):
            npi_d = np.pi*k*(d - a)/(b-a)
            npi_c = np.pi*k*(c - a)/(b-a)
            val_one = (np.cos(npi_d)*np.exp(d) - np.cos(npi_c)*np.exp(c))
            val_two = (k*np.pi/(b-a))*(np.sin(npi_d)*np.exp(d)-np.sin(npi_c)*np.exp(c))
            return (val_one + val_two)/(1+(k*np.pi/(b-a))**2)

        def psi_n(a,b,c,d,k):
            if np.all(k==0):
                return d-c
            else:
                return ((b-a)/(k*np.pi))*(np.sin(k*np.pi*(d - a)/(b-a))-np.sin(k*np.pi*(c-a)/(b-a)))

        def H_k(a,b,k):
            if self.optionType == 1:
                c = 0
                d = b
            elif self.optionType == -1:
                c = a
                d = 0
            return (self.optionType*2/(b-a))*(upsilon_n(a,b,c,d,k) - psi_n(a,b,c,d,k))

        if self.strike is not np.array:
            self.strike = np.array(self.strike).reshape([len(self.strike),1])

        x0 = np.array([np.log(self.S0/i) for i in self.strike])
        k = np.array([i for i in np.arange(1,N)])
        price = np.empty((self.strike.shape[0],N))
        price[:,0:1] = H_k(a,b,0)*self.Heston_Char_func(0)/2

        price[:,1:] = H_k(a,b,k)*self.Heston_Char_func(k*np.pi/(b-a))*np.exp(1j*k*np.pi*(x0 - a)/(b-a))

        return np.diag(np.sum(price.real, axis = 1)*np.exp(-self.rfr*self.maturity)*self.strike)

#TODO: Correct the fourier method. Chnage the adjusted characterstic function in Heston

    def price_FourierMethod(self,t_max,N):
        delta_t = t_max/N
        from_1_to_N = np.linspace(1,N,N)
        t_n = (from_1_to_N - 1/2)*delta_t
        price = dict()
        for strike in self.strike:
            k_log = np.log(self.S0/strike)
            first_int = np.sum((((np.exp(-1j*t_n*k_log)*self._adj_Heston_Char_func(t_n)).imag/t_n)*delta_t),axis=0)
            second_int = np.sum((((np.exp(-1j*t_n*k_log)*self.Heston_Char_func(t_n)).imag/t_n)*delta_t),axis=0)
            if self.optionType == 1:
                price[str(strike)] = (self.S0*(1/2 + first_int/np.pi)-np.exp(-self.rfr*self.maturity)*strike*(1/2 + second_int/np.pi))
            else:
                price[str(strike)] = (np.exp(-self.rfr*self.maturity)*strike*(1/2 - second_int/np.pi)-
                                 self.S0*(1/2-first_int/np.pi))
        return price



def plot_surface(data, x_axis,y_axis,x_label = "Xlabel",y_label="Ylabel", z_label = "Zlabel",
                 title="title",save=False,filePath=None,fsize=(8,8),show=False):
    fig = plt.figure(figsize=fsize)
    ax = Axes3D(fig)
    X, Y = np.meshgrid(x_axis,y_axis)
    ax.plot_surface(X, Y, data, cmap='coolwarm', linewidth=0, antialiased=False)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    if save:
        if filePath is None:
            print("Input filepath to save figure")
        fig.savefig(filePath)
    if show:
        plt.show()

if __name__ == "__main__":
    import time
    import sys
    # Heston Parameters
    v0 = 0.06
    kappa = 9
    vbar = 0.06
    gamma = 0.5
    rho = -0.4

    # Option Parameters
    S0 = 100
    r = 0.03
    T = 0.5
    N = 10000
    # K = np.array([95,100,105])
    K = np.array([90, 95, 100, 105, 110])

    # c1 = 0
    # c2 = T
    # c4 = 0
    lst_L = [5, 8, 10, 12, 15, 20, 25, 30]
    # a = c1 - L*np.sqrt(c2 - np.sqrt(c4)) # we use +/- L*sqrt(T)
    # b = c1 + L*np.sqrt(c2 - np.sqrt(c4))
    start = time.time()
    for L in lst_L:
        loopstart = time.time()
        years = 25  # Farthest maturity
        mat = [7 / 360, 14 / 360, 30 / 360, 60 / 360, 0.5]
        mat.extend([i for i in range(1, years + 1)])
        HestonPrices = np.empty([K.shape[0], len(mat)])
        BS_IVs = np.empty_like(HestonPrices)
        for i in range(0, len(mat)):
            T = mat[i]
            option = EuroOption(S0, K, np.sqrt(v0), r, T)  # setting sigma(IV)^2 = v0
            option.initializeHestonParameters(kappa, vbar, gamma, rho, v0, S0, r, T)
            a = -L * np.sqrt(T)
            b = L * np.sqrt(T)
            prices = option.price_Heston_CosMethod(a, b, N)
            HestonPrices[:, i] = prices
            BS_IVs[:, i] = option._getBS_ImpVol(prices)

        # Check if the BS analytical price matched using computed IV's
        BS_analyticalPrice = np.empty_like(HestonPrices)
        for i in range(0, len(mat)):
            T = mat[i]
            sigma_new = BS_IVs[:, i]
            option2 = EuroOption(S0, K, sigma_new, r, T)
            BS_analyticalPrice[:, i] = option2.price_BS_analytical()

        diff = HestonPrices - BS_analyticalPrice
        path_surface = outputPath + "BSImpVol_Surface_N{}k_L{}".format(int(N / 1000), L)
        path_diff = outputPath + "HestonVsBS_Diff_Surface_N{}k_L{}".format(int(N / 1000), L)
        path_priceSurface = outputPath + "HestonPrice_Surface_N{}k_L{}".format(int(N / 1000), L)
        plot_surface(BS_IVs.T, K, mat, x_label="Strikes", y_label="Time to Maturity", z_label="BS Imp Vol",
                     title="BS ImpVol Surface", save=True, filePath=path_surface+".png")
        plot_surface(diff.T, K, mat, x_label="Strikes", y_label="Time to Maturity", z_label="Diff",
                     title="Price Difference", save=True, filePath=path_diff+"_priceDiff.png")
        plot_surface(HestonPrices.T, K, mat, x_label="Strikes", y_label="Time to Maturity", z_label="BS Imp Vol",
                     title="Heston Surface", save=True, filePath=path_priceSurface+".png")

        cols = ["1w","2w","1M","3M","6M"]
        cols.extend(["{}Y".format(i) for i in range(1, years + 1)])
        df_BS = pd.DataFrame(data=BS_IVs,
                             index=K,
                             columns=cols)
        df_diff = pd.DataFrame(data=diff,
                             index=K,
                             columns=cols)
        df_HestonPrices = pd.DataFrame(data=HestonPrices,
                             index=K,
                             columns=cols)
        writer = pd.ExcelWriter(path_surface+".xlsx", engine='xlsxwriter')
        workbook = writer.book
        worksheet1 = workbook.add_worksheet('BS_ImpVolSurface')
        worksheet2= workbook.add_worksheet('diff_HestonVsBSPrice')
        worksheet3 = workbook.add_worksheet('HestonPrices')
        writer.sheets['BS_ImpVolSurface'] = worksheet1
        writer.sheets['diff_HestonVsBSPrice'] = worksheet2
        writer.sheets['HestonPrices'] = worksheet3

        df_BS.to_excel(writer, sheet_name='BS_ImpVolSurface', startrow=0, startcol=0)
        df_diff.to_excel(writer, sheet_name='diff_HestonVsBSPrice', startrow=0, startcol=0)
        df_HestonPrices.to_excel(writer, sheet_name='HestonPrices', startrow=0, startcol=0)
        writer.save()
    end = time.time()
    print("Total runtime: {:.5f}s".format((end-start)))
# fig1, ax1 = plt.subplots()
# for i in range(9,len(mat)):
#     ax1.plot(K,data[:,i],label = "maturity {} years".format(mat[i]))
# ax1.grid()
# ax1.legend()
# #ax.set_xticks(np.arange(0, 50001, 5000))
# ax1.set_xlabel("Strikes")
# ax.set_ylabel("BS IV")
# ax.set_title("IV")
# plt.show()

# print(data[:,9])
# print(diff[:,9])