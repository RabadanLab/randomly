# -*- coding: utf-8 -*-

"""Main module."""

import pandas as pd
import numpy as np
from scipy import stats, linalg

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import seaborn as sns

import warnings


from .visualization import Visualize
from .clustering import Cluster


class Rm(Visualize, Cluster):
    """Random Matrix Theory Analysis of Principal Components
    Parameters
    ----------
    eigen_solver: string {'wishart'}. Find eigenvalue and eigenvectors
        wishart: Compute wishart matrix and find eigenvalues and eigenvectors
        of the Wishart matrix

    tol: float >= 0, optional (default .0)
        Tolerance for singular values computed by svd_solver == 'arpack'.
    preprocessing: string {'sc', 'False'}
            sc: run single cell preprocessing
            False: skip preprocessing

    random_states: int, RandomState instance or None, optional (default None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    components_: array, shape (n_components, n_genes)
        Principal axes in feature space, representing the directions of
        maximum variance in the data. The components are sorted by
        ``explained_variance_``.

    explained_variance_: array, shape (n_components,)
        The amount of variance explained by each of the selected components.
        Equal to n_components largest eigenvalues
        of the covariance matrix of X.

    explained_variance_ratio_: array, shape (n_components,)
        Percentage of variance explained by each of the selected components.
        If ``n_components`` is not set then all components are stored and the
        sum of explained variances is equal to 1.0.

    mean_: array, shape (n_features,)
        Per-gene empirical mean, estimated from the original set.

    noise_variance_: float
        The estimated noise covariance

    cell_names: list of cell names

    gene_names: list of gene names

    X: 2D array
        Preprocessed single cell matrix

    n_cells: int
    number of cells

    n_genes: int
     number of genes

    L: list
        Eigenvalues of the Wishart matrix
    V: 2D array
        Eigenvectors of the Wishart matrix

    Lr: list
        Eigenvalues of the Wishart randomized matrix

    Vr: 2D array
        Eigenvectors of the Wishart randomized matrix

        L_mp: list
        Estimated Marchenko - Pastur Eigenvalues

    lambda_c: float
         Critical eigenvalues estimated by Tracy - Widom distribution

    n_components: int
        Number of components above the Marchenko - Pastur distribution

    gamma: float
            Estimated gamma of the Marchenko - Pastur distribuiton

    sigma: float
        Estimated sigma of the Marchenko - Pastur distribution

    b_plus: float
        Estimated upper bound of the Marchenko - Pastur distribution

    b_minus: float
        Estimated lower bound of the Marchenko - Pastur distribution
    """


    
    
    def __init__(self, tol=0.0, random_state=None):
        self.eigen_solver = 'wishart'
        self.tol = tol
        self.random_state = random_state
        self.n_cells = 0
        self.n_genes = 0
        self.cell_names = []
        self.gene_names = []
        self.normal_genes = []
        self.normal_cells = []
        self.L = []
        self.V = None
        self.X = None
        self.X2 = None
        self.X3 = None
        self.Lr = []
        self.Vr = None
        self.explained_variance_ = []
        self.total_variance_ = []
        self.L_mp = []
        self.lambda_c = 0
        self.n_components = 0
        self.gamma = 0
        self.p = 0
        self.sigma = 0
        self.x_peak = 0
        self.b_plus = 0
        self.b_minus = 0
        self._preprocessing_flag = False
        self._selection_flag = True
        self._preprocessing2_flag = False
        self.X_vis=None

    def style(self):
        params = {'legend.fontsize': 'medium',
         'axes.labelsize': 'large',
         'axes.titlesize':'medium',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium',
         'figure.dpi': 100,
         'axes.linewidth': 0.5,
         }
        np.seterr(invalid='ignore')
        sns.set_style("white")
        plt.rcParams.update(params)
        
    def style_mp_stat(self):
        plt.style.use("ggplot")
        np.seterr(invalid='ignore')
        sns.set_style("white")
        sns.set_context("paper")
        sns.set_palette("deep")
        plt.rcParams['axes.linewidth'] =0.5
        plt.rcParams['figure.dpi'] = 100
        
        
    def fit(self, df=None, eigen_solver='wishart'):
        """Fit RM model

        Parameters
        ----------
        df: Pandas dataframe, shape (n_cells, n_genes)
            where n_cells in the number of cells
            and n_genes is the number of genes.
        -------
        self: object
            Returns the instance itself.
        """
        if self._preprocessing_flag:
            print("Preprocessed data is being used for fitting")
        else:
            print('Data has not been preprocessed')
            detected_genes = df.columns[(np.sum(df.values, axis=0) > 0)]
            detected_cells = df.index[(np.sum(df.values, axis=1) > 0)]
            df = df.loc[detected_cells, detected_genes]
            self.X = df.values
            self.n_cells = df.shape[0]
            self.n_genes = df.shape[1]
            self.normal_genes = df.columns
            self.normal_cells = df.index
            self.X3 = pd.DataFrame(self.X, index=self.normal_cells,
                columns=self.normal_genes)
        self._fit()
        return

    def _fit(self):
        """Fit the model for the dataframe df and apply
           the dimensionality reduction by removing the eigenvalues
           that follow Marchenko - Pastur distribution
        """
        self.mean_ = np.mean(self.X, axis=0)
        self.std_ = np.std(self.X, axis=0, ddof=0)
        self.X = (self.X-self.mean_) / (self.std_+0.0)

        """Dispatch to the right submethod depending on
           the chosen solver"""
        if self.eigen_solver == 'wishart':
            Y = self._wishart_matrix(self.X)
            (self.L, self.V) = self._get_eigen(Y)
            Xr = self._random_matrix(self.X)
            Yr = self._wishart_matrix(Xr)
            (self.Lr, self.Vr) = self._get_eigen(Yr)

            self.explained_variance_ = (self.L**2) / (self.n_cells)
            self.total_variance_ = self.explained_variance_.sum()

            self.L_mp = self._mp_calculation(self.L, self.Lr)
            self.lambda_c = self._tw()
            self.peak = self._mp_parameters(self.L_mp)['peak']
        else:
            print('''Solver is undefined, please use
                     Wishart Matrix as eigenvalue solver''')


        self.Ls = self.L[self.L > self.lambda_c]
        Vs = self.V[:, self.L > self.lambda_c]
  
        self.Vs = Vs
        noise_boolean = ((self.L < self.lambda_c) & (self.L > self.b_minus))
        Vn = self.V[:, noise_boolean]
        self.Ln = self.L[noise_boolean]
        self.n_components = len(self.Ls)

        Vna = self.Vr[:, len(self.Lr)//2 
                 - self.n_components//2: len(self.Lr)//2
                 + self.n_components//2
                 + (self.n_components) % 2]

        signal_projected_genes = self._project_genes(self.X, Vs)
        random_projected_genes = self._project_genes(self.X, Vna)

        noise_left_projected_genes = self._project_genes(
            self.X,  Vn[:, :self.n_components])
        noise_right_projected_genes = self._project_genes(
            self.X, Vn[:, -self.n_components:])
        noise_projected_genes = self._project_genes(self.X,  Vn)

        # print(noise_right_projected_genes)
        self._s = np.square(signal_projected_genes).sum(axis=1)
        self._sa = np.square(random_projected_genes).sum(axis=1)
        self._snl = np.square(noise_left_projected_genes).sum(axis=1)
        self._snr = np.square(noise_right_projected_genes).sum(axis=1)
        
        self.components_genes = dict()
        for j in range(self.n_components):
            self.components_genes[j] = np.array(self.normal_genes)[
                np.square(signal_projected_genes[:, -j - 1])
                > 10 * np.max(np.square(noise_projected_genes),
                              axis=1)
            ]

        self.X = np.dot(np.dot(Vs, Vs.T),
                        self.X)  
        # X = U S V^T= U(V S)^T = U (X^T U)^T = U U^T X ~ Us Us^T X

    def return_cleaned(self, 
                        fdr=None, 
                        sample_variance=None,
                        path=False):
        ''' Method returns the dataframe with denoised single
        cell data if fdr == True, return method returns structure
        genes up to the fdr level

        Parameters
        ----------
        path: string
                Path to save the plot
        fdr_cut: float

        Returns
        -------
        object: Pandas DataFrame shape(n_cells, n_genes)
                Cleaned matrix
        '''
        self._selection_flag = True
        if (fdr == 1) or (sample_variance==0):
             return self.X3
        else:
            genes = self.select_genes(fdr, sample_variance)
            if self._selection_flag:
                if path:
                    pd.DataFrame(self.X, index=self.normal_cells, 
                        columns=self.normal_genes).loc[:, genes]\
                        .applymap(lambda x: round(x,3)).to_csv(path, sep='\t')
                return pd.DataFrame(self.X,index=self.normal_cells,
                                    columns=self.normal_genes).loc[:, genes]
            else:
                print('''Criterium for selection of genes undefined,
                     please select FDR < 1 OR sample variance > 0''')
                            
    def _to_tpm(self, X):
        '''Transform transcripts to transcripts per million'''
        return np.transpose(np.transpose(X) /
                            np.sum(X, axis=1) * 10**(6)
                            )

    def _tw(self):
        '''Tracy-Widom critical eignevalue'''
        gamma = self._mp_parameters(self.L_mp)['gamma']
        p = len(self.L) / gamma
        sigma = 1 / np.power(p, 2/3) * np.power(gamma, 5/6) * \
            np.power((1 + np.sqrt(gamma)), 4/3)
        lambda_c = np.mean(self.L_mp) * (1 + np.sqrt(gamma)) ** 2 + sigma
        self.gamma = gamma
        self.p = p
        self.sigma = sigma
        return lambda_c

    def _wishart_matrix(self, X):
        """Compute Wishart Matrix of the cells"""
        return np.dot(X, X.T) / X.shape[1]

    def _random_matrix(self, X):
        return np.apply_along_axis(np.random.permutation, 0, X)

    def _get_eigen(self, Y):
        """Compute Eigenvalues of the real symmetric matrix"""
        (L, V) = linalg.eigh(Y)
        return (L, V)

    def _mp_parameters(self, L):
        """Compute Parameters of the Marchenko
        Pastur Distribution of eigenvalues L"""
        moment_1 = np.mean(L)
        moment_2 = np.mean(np.power(L, 2))
        gamma = moment_2 / float(moment_1**2) - 1
        s = moment_1
        sigma = moment_2
        b_plus = s * (1 + np.sqrt(gamma))**2
        b_minus = s * (1 - np.sqrt(gamma))**2
        x_peak = s * (1.0-gamma)**2.0 / (1.0+gamma)
        dic = {'moment_1': moment_1,
               'moment_2': moment_2,
               'gamma': gamma,
               'b_plus': b_plus,
               'b_minus': b_minus,
               's': s,
               'peak': x_peak,
               'sigma': sigma
               }
        return dic

    def _marchenko_pastur(self, x, dic):
        '''Distribution of eigenvalues'''
        pdf = np.sqrt((dic['b_plus'] - x) * (x-dic['b_minus']))\
            / float(2 * dic['s'] * np.pi * dic['gamma'] * x)
        return pdf

    def _mp_pdf(self, x, L):
        '''Marchnko-Pastur PDF'''
        vfunc = np.vectorize(self._marchenko_pastur)
        y = vfunc(x, self._mp_parameters(L))
        return y

    def _mp_calculation(self, L, Lr, eta=1, eps=10**-6, max_iter=1000):
        converged = False
        iter = 0
        loss_history = []
        b_plus = self._mp_parameters(Lr)['b_plus']
        b_minus = self._mp_parameters(Lr)['b_minus']
        L_updated = L[(L > b_minus) & (L < b_plus)]
        new_b_plus = self._mp_parameters(L_updated)['b_plus']
        new_b_minus = self._mp_parameters(L_updated)['b_minus']
        while not converged:
            loss = (1 - float(new_b_plus) / float(b_plus))**2
            loss_history.append(loss)
            iter += 1
            if loss <= eps:
                converged = True
            elif iter == max_iter:
                print('Max interactions exceeded!')
                converged = True
            else:
                gradient = new_b_plus - b_plus
                new_b_plus = b_plus + eta * gradient
                L_updated = L[(L > new_b_minus) & (L < new_b_plus)]
                b_plus = new_b_plus
                b_minus = new_b_minus
                new_b_plus = self._mp_parameters(L_updated)['b_plus']
                new_b_minus = self._mp_parameters(L_updated)['b_minus']
        self.b_plus = new_b_plus
        self.b_minus = new_b_minus
        return L[(L > new_b_minus) & (L < new_b_plus)]

    def _project_genes(self, X, V):
        '''Return (n_genes, n_components) matrix
           of gene projections on components'''
        return np.dot(X.T, V)

    def _project_cells(self, X, V):
        '''Return (n_cells, n_components) matrix
           of cell projections on components'''
        return np.dot(X, np.dot(X.T, V))

    def _get_gene_norm(self, X):
        return np.sqrt(np.square(X).sum(axis=0) / X.shape[0])

    def plot_mp(self, comparison=True, path=False,
                info=True, bins=None, title=None):
        """Plot Eigenvalues,  Marchenko - Pastur distribution,
        randomized data and estimated Marchenko - Pastur for
        randomized data

        Parameters
        ----------
        path: string
                Path to save the plot
        fit: boolean
            The data.
        fdr_cut: float

        Returns
        -------
        object: plot
        """

        self.style_mp_stat()
        if bins is None:
            #bins = self.n_cells
            bins= int(self.n_cells/3.0)
        x = np.linspace(0, int(round(np.max(self.L_mp) + 0.5)), 2000)
        y = self._mp_pdf(x, self.L_mp)
        yr = self._mp_pdf(x, self.Lr)

        if info:
            fig = plt.figure(dpi=100)
            fig.set_tight_layout(False)

            ax = fig.add_subplot(111)
        else:
            plt.figure(dpi=100)

        plot = sns.distplot(self.L,
                            bins=bins,
                            norm_hist=True,
                            kde=False,
                            hist_kws={"alpha": 0.85,
                                      "color": sns.xkcd_rgb["cornflower blue"]
                                      }
                            )

        plot.set(xlabel='First cell eigenvalues normalized distribution')
        plt.plot(x, y,
                 sns.xkcd_rgb["pale red"],
                 lw=2)

        
        MP_data = mlines.Line2D([], [], color=sns.xkcd_rgb["pale red"],
                            label='MP for random part in data',linewidth=2)
        MP_rand = mlines.Line2D([], [], color=sns.xkcd_rgb["sap green"],
                            label='MP for randomized data' ,linewidth=1.5,
                            linestyle='--')
        randomized=mpatches.Patch(color=sns.xkcd_rgb["apple green"], 
                            label='Randomized data',alpha=0.75,
                            linewidth=3, fill=False)
        data_real=mpatches.Patch(color=sns.xkcd_rgb["cornflower blue"], 
                            label='Real data',alpha=0.85)
        
        
        
        if comparison:
            sns.distplot(self.Lr, bins=30, norm_hist=True,
                         kde=False,
                         hist_kws={"histtype": "step", "linewidth": 3,
                                   "alpha": 0.75,
                                   "color": sns.xkcd_rgb["apple green"]}
                         )

            plt.plot(x, yr,
                     sns.xkcd_rgb["sap green"],
                     lw=1.5,
                     ls='--'
                     )
                        
            plt.legend(handles=[data_real, MP_data , randomized,MP_rand], 
                       loc="upper right", frameon=True)

        else:            
            plt.legend(handles=[data_real, MP_data], 
                        loc="upper right",frameon=True)

        plt.xlim([0, int(round(max(np.max(self.Lr), np.max(self.L_mp))
                               + 1.5))])
        plt.grid(b=True, linestyle='--',lw=0.3)
        
        if title:
            plt.title(title)

        if info:
            dic = self._mp_parameters(self.L_mp)
            info1 = (r'$\bf{Data Parameters}$' + '\n{0} cells\n{1} genes'
                     .format(self.n_cells, self.n_genes))
            info2 = ('\n' + r'$\bf{MP\ distribution\ in\ data}$'
                     + '\n$\gamma={:0.2f}$ \n$\sigma^2={:1.2f}$\
                      \n$b_-={:2.2f}$\n$b_+={:3.2f}$'
                     .format(dic['gamma'], dic['s'], dic['b_minus'], 
                     dic['b_plus']))
            info3 = ('\n' + r'$\bf{Analysis}$' +
                     '\n{0} eigenvalues > $\lambda_c (3 \sigma)$\
                     \n{1} noise eigenvalues'
                     .format(self.n_components, self.n_cells - \
                      self.n_components))

            
            ks=stats.kstest(self.L_mp, self._call_mp_cdf(self.L_mp,dic))
            
            info4= '\n'+r'$\bf{Statistics}$'+'\nKS distance ={0}'\
                                .format(round(ks[0], 4))\
                                +'\nKS test p-value={0}'\
                                .format(round(ks[1], 2))
        
            infoT= info1+info2+info4+info3
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            
            at = AnchoredText(infoT, loc=2, prop=dict(size=10),
                              frameon=True,
                              bbox_to_anchor=(1., 1.024),
                              bbox_transform=ax.transAxes)
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            lgd = ax.add_artist(at)

            if path:
                plt.savefig(path, bbox_extra_artists=(
                    lgd,), bbox_inches='tight')
        else:
            if path:
                plt.savefig(path)
        plt.show()
        return

    def plot_statistics(self, path=False, fit=True):
        """Plot statistics,

        Parameters
        ----------
        path: string
                Path to save the plot
        fit: boolean
            The data.
        fdr_cut: float

        Returns
        -------
        object: plot

        """
        
        self.style_mp_stat()       

        gs = gridspec.GridSpec(2, 2)
        fig = plt.figure(dpi=100, figsize=[9, 4.4])
        fig.set_tight_layout(False)
        ax = plt.subplot(gs[0, 0])
        ax0 = plt.subplot2grid((1, 3), (0, 0), colspan=2)

        sns.distplot(self._sa,
                     norm_hist=True,
                     kde=False,
                     bins=100,
                     hist_kws={"alpha": 0.8,
                               "color": sns.xkcd_rgb["golden yellow"],
                               "zorder": 1})
        sns.distplot(self._s,
                     norm_hist=True,
                     kde=False,
                     bins=200,
                     hist_kws={"alpha": 0.6,
                               "color": sns.xkcd_rgb["cerulean"],
                               "zorder": 3})
        sns.distplot(self._snl,
                     norm_hist=True,
                     kde=False,
                     bins=100,
                     hist_kws={"alpha": 0.55,
                               "color": sns.xkcd_rgb["leaf green"],
                               "zorder": 5})
        sns.distplot(self._snr,
                     norm_hist=True,
                     kde=False,
                     bins=100,
                     hist_kws={"alpha": 0.5,
                               "color": sns.xkcd_rgb["cerise"],
                               "zorder": 7})

        plt.xlim([0, 2 * np.max(self._snr)])
        plt.xlabel('Normalized sample variance')
        plt.ylabel('Sample variance probability distribution')

        hist_sa = mpatches.Patch(color=sns.xkcd_rgb["golden yellow"],
                                 label='{0} random vectors'.format(
                                     self.n_components),
                                 alpha=0.8)
        hist_s = mpatches.Patch(color=sns.xkcd_rgb["cerulean"],
                                label='All {0} signal eigenvectors'.format(
                                    self.n_components),
                                alpha=0.6)
        hist_snl = mpatches.Patch(color=sns.xkcd_rgb["leaf green"],
                                  label='Lowest {0} MP eigenvectors'.format(
                                      self.n_components),
                                  alpha=0.55)
        hist_snr = mpatches.Patch(color=sns.xkcd_rgb["cerise"],
                                  label='Largest {0} MP eigenvectors'.format(
                                      self.n_components),
                                  alpha=0.5)

        xgr = np.linspace(0, np.max(self._snr)+200, 1000)
        y_fdr = np.vectorize(self._fdr)(xgr)

        if fit:
            xgl = np.linspace(0, np.max(self._snl), 1000)
            xk = np.linspace(0, np.max(self._sa), 1000)
            xs = np.linspace(0, np.max(self._s) + 0.0, 1000)

            fits = self._fit_gamma(self._s)
            fitl = self._fit_gamma(self._snl)
            fitr = self._fit_gamma(self._snr)
            ygl = self._gamma_pdf(xgl, fitl)
            ygr = self._gamma_pdf(xgr, fitr)
            ys = self._gamma_pdf(xs, fits)
            y = stats.chi2.pdf(xk, self.n_components)
            plt.ylim(0, max(y)+0.02)

            plt.plot(xk, y,
                     zorder=2,
                     color=sns.xkcd_rgb["adobe"],
                     linestyle='--',
                     linewidth=1.1
                     )
            plt.plot(xgl, ygl,
                     zorder=6,
                     color=sns.xkcd_rgb["grassy green"],
                     linestyle='-',
                     linewidth=1.5,
                     )
            plt.plot(xgr, ygr,
                     zorder=8,
                     color=sns.xkcd_rgb["rose red"],
                     linestyle='-',
                     linewidth=1.5
                     )
            plt.plot(xs, ys,
                     zorder=4,
                     color=sns.xkcd_rgb["blue blue"],
                     linestyle='-',
                     linewidth=1.5
                     )
            line_gammal = mlines.Line2D([], [],
                                        color=sns.xkcd_rgb["grassy green"],
                                        label=r'Gamma PDF: $\alpha ={:0.1f}$, $\beta = {:1.1f}$'
                                        .format(fitl[0], 1/fitl[2]),
                                        linewidth=1.5)
            line_gammar = mlines.Line2D([], [],
                                        color=sns.xkcd_rgb["rose red"],
                                        label=r'Gamma PDF: $\alpha ={:0.1f}$,$\beta ={:1.1f}$'
                                        .format(fitr[0], 1 / fitr[2]),
                                        linewidth=1.5)
            line_chi = mlines.Line2D([], [],
                                     color=sns.xkcd_rgb["adobe"],
                                     label='Chi-Squared Distribution',
                                     linewidth=1.1,
                                     linestyle='--')
            line_gammas = mlines.Line2D([], [],
                                        color=sns.xkcd_rgb["blue blue"],
                                        label=r'Gamma PDF: $\alpha =%.1f$,$\beta = %.2f$'\
                                        % (fits[0], 1/fits[2]),
                                        linewidth=1.5)
            plt.legend(handles=[hist_s, line_gammas, hist_snr, 
                                line_gammar, hist_snl,
                                line_gammal, hist_sa, line_chi],
                       title=r'$\bf{Gene\ projection\ samples}$',
                       loc="upper right",
                       frameon=True)
        else:
            plt.legend(handles=[hist_s, hist_snr, hist_snl, hist_sa],
                       title=r'$\bf{Gene\ projection\ samples}$',
                       loc="upper right",
                       frameon=True)

        ax1 = plt.subplot2grid((1, 3), (0, 2))
        host = ax1
        par = ax1.twinx()

        host.set_ylabel("False Discovery Rate")
        par.set_ylabel("Number of genes")

        p1, = host.plot(xgr, y_fdr[0],
                        label="False Discovery Rate",
                        ls='--',
                        lw=2,
                        color=sns.xkcd_rgb["pumpkin orange"])
        p2, = par.plot(xgr, y_fdr[1],
                       label="Number of genes",
                       ls='-.',
                       lw=1.5,
                       color=sns.xkcd_rgb["violet blue"])

        host.yaxis.get_label().set_color(p1.get_color())
        par.yaxis.get_label().set_color(p2.get_color())

        line_fdr = mlines.Line2D([], [],
                                 color=sns.xkcd_rgb["pumpkin orange"],
                                 label='Ratio signal vs \nlargest {0} MP\ndistributions'
                                 .format(self.n_components),
                                 linewidth=2,
                                 linestyle='--')
        line_genes = mlines.Line2D([], [],
                                   color=sns.xkcd_rgb["violet blue"],
                                   label='Relevant genes',
                                   linewidth=1.5,
                                   linestyle='-.')

        ax1.legend(handles=[line_fdr, line_genes],
                   loc="upper right",
                   frameon=True)
        host.set_yscale("log")
        host.grid(b=True, linestyle='--',lw=0.3)
        host.set_xlabel('Normalized sample variance')
        host.set_ylim(min(y_fdr[0][np.nonzero(y_fdr[0])]),1.2)
        host.set_xlim(-2, max(xgr))

        plt.tight_layout()

        if path:
            plt.savefig(path)
        return plt.show()

    def _fdr(self, x):
        '''Number of genes for the false discovery rate'''
        area_noise = len(self._snr[self._snr > x])
        area_signal = len(self._s[self._s > x])
        if area_signal>0:
            fdr_x = area_noise/float(area_signal)
            genes = self._s[self._s > x].tolist()
        else:
            fdr_x=0
            genes=[]
        return (fdr_x, len(genes))

    def _fdr_genes(self, x):
        '''Number of signal genes for the false discovery rate'''
        area_noise = len(self._snr[self._snr > x])
        area_signal = len(self._s[self._s > x])
        fdr_x = area_noise / float(area_signal)
        # genes = self._s[self._s > x].tolist()
        return (fdr_x, x)

    def select_genes(self, fdr=None, sample_variance=None):
        if fdr is not None and fdr < 1.0 and sample_variance is None: 
            xgr = np.linspace(0, np.max(self._snr), 1000)
            y_fdr = np.vectorize(self._fdr_genes)(xgr)
            idx = np.abs(y_fdr[0] - fdr).argmin()
            x = y_fdr[1][idx]
            genes = np.array(self.normal_genes)[self._s > x].tolist()
        elif sample_variance is not None and \
        sample_variance > 0.0 and fdr is None:
            genes = np.array(self.normal_genes)\
            [self._s > sample_variance].tolist()
        else:
            self._selection_flag = False
            genes = np.array(self.normal_genes).tolist()
        return genes
    

    def _fit_gamma(self, x):
        alpha, loc, beta = stats.gamma.fit(x, loc=0, scale=1)
        return (alpha, loc, beta)

    def _gamma_pdf(self, x, fits):
        alpha, loc, beta = fits
        y = stats.gamma(a=alpha, loc=loc, scale=beta).pdf(x)
        return y

    def _cdf_marchenko(self,x,dic):
        if x < dic['b_minus']: 
            return 0.0
        elif x>dic['b_minus'] and x<dic['b_plus']:
            return 1/float(2*dic['s']*np.pi*dic['gamma'])*\
            float(np.sqrt((dic['b_plus']-x)*(x-dic['b_minus']))+\
            (dic['b_plus']+dic['b_minus'])/2*np.arcsin((2*x-dic['b_plus']-\
            dic['b_minus'])/(dic['b_plus']-dic['b_minus']))-\
        np.sqrt(dic['b_plus']*dic['b_minus'])*np.arcsin(((dic['b_plus']+\
            dic['b_minus'])*x-2*dic['b_plus']*dic['b_minus'])/\
        ((dic['b_plus']-dic['b_minus'])*x)) )+np.arcsin(1)/np.pi
        else:
            return 1.0
  
    def _call_mp_cdf(self,L,dic):
        "CDF of Marchenko Pastur"
        func= lambda y: list(map(lambda x: self._cdf_marchenko(x,dic), y))
        return func

    def _shapiro(self,x):  
        return stats.shapiro(x)[0]
    
    def refining(self,min_trans_per_gene=15):
        if not self._preprocessing2_flag:
            raise ValueError('''Cleaning function only works with 
                                after running preprocess function
                                     with option refined=True''')
        #low-expressed genes
        genesproj=np.dot(self.X3.T,(abs(self.Vr))[:,:]).T 
        warnings.filterwarnings('once')
        shapiro_stat_gen=np.apply_along_axis(self._shapiro, 0, genesproj[:,:])
        warnings.filterwarnings('ignore')
        dist_genes=pd.Series(1-shapiro_stat_gen,index=self.X2.columns)
        loop=0
        k=0
        while(loop==0):
            dist_genes_thrs=dist_genes.quantile([0.75-k]).values[0]
            genes_low_exp=dist_genes[(dist_genes>dist_genes_thrs)]\
            .index.tolist()
            self.X=self.X2.drop(genes_low_exp,axis=1)
            if self.X.sum().min()>min_trans_per_gene:
                loop=100
            else:
                k+=0.01
                               
        self.X=self.X.loc[(self.X.T>0).sum()>0]    # filter out cells without any expression after cleaning
        self.X=self.X.loc[:,self.X.sum()>0]       # filter out genes without any expression after cleaning
                
        self.normal_genes = self.X.columns
        self.normal_cells = self.X.index
        self.X = np.log2(1 + self._to_tpm(self.X.values))
        self.n_cells = self.X.shape[0]
        self.n_genes = self.X.shape[1]
        
        self.X3 = pd.DataFrame(np.log2(1 + self._to_tpm(self.X2.values)), 
                            index=self.X3.index, 
                            columns=self.X3.columns).loc[self.normal_cells,:]   
        
        print(self.n_cells, ' cells and ', self.n_genes, ' genes')
        self._preprocessing_flag = True
    
    def preprocess(self, df,
                   min_tp=0,
                   min_genes_per_cell=0,
                   min_cells_per_gene=0,
                  refined=False):
        """The method executes preprocessing of the data by removing
        1. Genes and cells that have less than 100 transcripts by default.
        2. Cells are removed that express less than min_tp genes
        3. Genes are removed that expressed less than in 10 cells

        4. Transcripts are being converted to log2(1+TPM).
        5. Genes are  standard-normalized to have zero mean and standard
        deviation equal to 1.

        Input: Pandas DataFrame, shape(n_cells, n_genes)

        Parameters
        ----------
        df: Pandas dataframe, shape (n_cells, n_genes)
            where n_cells in the number of cells
            and n_genes is the number of genes.

        min_tp: float
            minimum number of transcripts to be observed
            in every cell and every gene
        Returns
        -------
        self: self
            Returns the instance itself.
        """

        if self._preprocessing_flag:
            print(
                '''Single Cell data has already been preprocessed 
                with method preprocess''')
        else:
            # Duplicated gene and cell names are removed
            if not df.index.is_unique:
                print('''Cell names are not unique. Cell names are reset''')
                df.index = range(len(df.index))

            if not df.columns.is_unique:
                print('''Gene names are not unique. Duplicated 
                    genes will be removed''')
                df = df.loc[:, ~df.columns.duplicated()]
            self.gene_names = df.columns.tolist()
            self.cell_names = df.index.tolist()

            self.normal_genes = df.columns[(np.sum(df.values, axis=0) > min_tp)\
            & (np.count_nonzero(df.values, axis=0) >= min_cells_per_gene)]

            self.normal_cells = df.index[(np.sum(df.values, axis=1) > min_tp) &\
            (np.count_nonzero(df.values, axis=1) >= min_genes_per_cell)]

            self.filtered_genes = list(
                set(self.gene_names) - set(self.normal_genes))
            df = df.loc[self.normal_cells, self.normal_genes]
            
            
            if refined:           
                sortgenes = df.mean().sort_values()     
                df = df[sortgenes.index.tolist()]     # sort genes by expression in transcripts

                self.X2 = df
                self.X3 = np.log2(1 + self._to_tpm(df.values))

                self.mean_ = np.mean(self.X3, axis=0)
                self.std_ = np.std(self.X3, axis=0, ddof=0)
                self.X3 = (self.X3-self.mean_) / (self.std_+0.0)

                self.X3 = pd.DataFrame(self.X3, index=self.normal_cells,
                 columns=df.columns)
                
                Xr = self._random_matrix(self.X3)
                Yr = self._wishart_matrix(Xr)
                (self.Lr, self.Vr) = self._get_eigen(Yr)
                print('Run the refining function')
                self._preprocessing2_flag = True
                
            else:
                self.X = np.log2(1 + self._to_tpm(df.values))
                self.n_cells = self.X.shape[0]
                self.n_genes = self.X.shape[1]
                self.X3 = pd.DataFrame(self.X, index=self.normal_cells,
                 columns=self.normal_genes)
                print(self.n_cells, ' cells and ', self.n_genes, ' genes')
                self._preprocessing_flag = True
