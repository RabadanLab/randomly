
"""
Visualization of Marchenko Pastur algorithm and implementation
PCA, tSNE dimensionality reduction techniques and gene visualization functions
"""
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.decomposition import PCA

import multiprocessing 
from MulticoreTSNE import MulticoreTSNE
import numpy as np
import pandas as pd
import seaborn as sns

from collections import defaultdict
from scipy.cluster import hierarchy
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

from .palettes import pallete_50


class Visualize():
    '''class for embedding visualization'''
    def __init__(self):
        self.X=None
        self.position_flag=False
            
    def fit_tsne(self,
                perplexity=30,
                learning_rate =1000,
                early_exaggeration=12,
                metric='correlation',
                n_comp=False,
                multicore=True,
                fdr=None,sample_variance=None): 
        """
        Embedding of single cell data with t-distributed stochastic neighborhood 
        embedding (tSNE) for 2D visualization. By default, we use Multicore-tsne 
        implementation by Dmitry Ulyanov 
        https://github.com/DmitryUlyanov/Multicore-TSNE>, 
        if number of processors is great than 1. Otherwise, scikit-learn 
        implementation is used. Default parametes by scikit-learn are used.

        Parameters
        ----------
        perplexity : float, optional (default: 30)

        The perplexity is related to the number of nearest neighbors that 
        is used in other manifold learning algorithms. Larger datasets usually 
        require a larger perplexity. Consider selecting a value between 5 and 50.
        The choice is not extremely critical since t-SNE is quite insensitive to 
        this parameter.

        early_exaggeration : 'float', optional (default: 12.0)
            Controls how tight natural clusters in the original space are in the
            embedded space and how much space will be between them. For larger
            values, the space between natural clusters will be larger in the
            embedded space. Again, the choice of this parameter is not very
            critical. If the cost function increases during initial optimization,
            the early exaggeration factor or the learning rate might be too high.

        learning_rate : 'float', optional (default: 1000)
            Note that the R-package "Rtsne" uses a default of 200.
            The learning rate can be a critical parameter. It should be
            between 100 and 1000. If the cost function increases during initial
            optimization, the early exaggeration factor or the learning rate
            might be too high. If the cost function gets stuck in a bad local
            minimum increasing the learning rate helps sometimes.

        Returns
        -------
        Updated insance self, with self.embedding containing 2D t-SNE coordianates
        """

        if self.X is None:
            raise ValueError('Nothing to plot, please fit the data first')
        else:
            self.selected_genes=self.select_genes(fdr,sample_variance)
            self.X_vis=self.X.copy()[:,self.normal_genes.isin(self.selected_genes)]

                            
        if not self._selection_flag:
            print('Criterium for selection of genes undefined, please select FDR < 1 OR sample variance > 0')
            print('      ')
        
        if n_comp:
            pca = PCA(n_components=n_comp, svd_solver='full')
            self.X_vis = pca.fit_transform(self.X_vis)

        n_jobs = multiprocessing.cpu_count()

        if n_jobs > 1 and multicore:
            tsne = MulticoreTSNE(n_jobs=n_jobs,
                                 init='random',
                                 metric=metric,
                                 perplexity=perplexity,
                                learning_rate=learning_rate,
                                early_exaggeration=early_exaggeration)

            print('computing t-SNE, using Multicore t-SNE for {0} jobs'.format(n_jobs))
            # need to transform to float64 for MulticoreTSNE...
            self.embedding = tsne.fit_transform(self.X_vis.astype('float64'))
        else:
            print('computing t-SNE, using scikit-learn implementation')
            tsne = manifold.TSNE(n_components=2,
                                  init='pca',
                                  random_state=0,
                                  metric='correlation',
                                  perplexity=perplexity)

            self.embedding = tsne.fit_transform(self.X_vis)
        print('atribute embedding is updated with t-SNE coordinates')
        if n_comp:
            self.X_vis=self.X.copy()[:,self.normal_genes.isin(self.selected_genes)]
        return
    
    def fit_pca(self, n_comp=2, fdr=None,sample_variance=None): 
        
        '''2D PCA of the Data based on first 2 principal components'''
        
        if not self._selection_flag:
            print('Criterium for selection of genes undefined, please select FDR < 1 OR sample variance > 0')
            print('      ')

        if self.X is None:
            raise ValueError('Nothing to plot, please fit the data first')
        else:
            self.selected_genes=self.select_genes(fdr,sample_variance)
            self.X_vis=self.X.copy()[:, self.normal_genes.isin(self.selected_genes)]
           
        pca = PCA(n_components=n_comp, svd_solver='full')
        self.embedding=pca.fit_transform(self.X_vis)
        print('atribute embedding is updated with t-SNE coordinates')
        return
            
    def plot(self,
           path=False,
           title=False,
           labels=False,
           palette=pallete_50,
           gene=False,
           data=False,
           size=4.5,
           fontsize=13.5,
           legend=False,
           legendcol=5,
           points=5,
           xytitle='t-SNE',
           tight=False
            ):
        "Ploting labels"
        
        fontsize=size*2.7
        if labels is not False:
           
            if not palette:
                palette=sns.color_palette("husl", len(set(labels))+1)
            
            if 0 in labels or 200001 in labels:
                palette=palette
            else:
                palette=self.palette_hierarchy
                
            with sns.plotting_context("paper", font_scale=1.5):
                self.style()
                g=sns.lmplot( x='x',
                            y='y',
                            fit_reg=False,
                            scatter_kws={'s': points,
                                         'alpha': 1},
                            hue='label',
                            data=pd.DataFrame(self.embedding,
                                              columns=  ['x','y'])    
                                              .join(pd.Series(labels, name='label')),
                            height=size,
                            palette=sns.set_palette(palette),
                            legend=False)
                g.set(yticks=[],xticks=[])

                if legend is not False:
                    if title:
                        sep=1.05
                    else:
                        sep=1.0
                    if type(legend) is list:
                        plt.legend(legend,
                                    loc='lower center',
                                    bbox_to_anchor=(0.5, sep),
                                    ncol=legendcol,
                                    frameon=True,
                                    markerscale=size//2,
                                    fontsize=size+3)     
                    elif legend is True:
                        plt.legend(loc='lower center', 
                                   bbox_to_anchor=(0.5, sep),
                                   ncol=legendcol,
                                   frameon=True, 
                                   markerscale=np.floor(size/2.),
                                   fontsize=size+3.0)
                        
                plt.xlabel(xytitle + '1', fontsize=fontsize)
                plt.ylabel(xytitle + '2', fontsize=fontsize)
                plt.autoscale(enable=True, axis='both', tight=tight)

        elif type(gene) is list:
                         
            if gene[0]=='library':
                color=(self.X3.T>0).sum()
            else:
                color=self.X3[gene].T.mean()
                
            self.style()
            fig=plt.figure(figsize=(size, size + 0.5), dpi=100)

            g=plt.scatter(self.embedding[:,0],
                          self.embedding[:,1],
                          s=points,
                          c=color,
                          alpha=1,
                          cmap='coolwarm'
                          )
            plt.xlabel(xytitle + '1', fontsize=fontsize)
            plt.ylabel(xytitle + '2', fontsize=fontsize)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            plt.autoscale(enable=True, axis='both',tight=tight)


            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes("right", "2.5%", pad="1%")

            if gene[0]=='library':
                plt.colorbar(g, cax=cax, label='library complexity')
            else:
                plt.colorbar(g, cax=cax, label='log2(1+TPM)')
                    
                
                
        elif type(gene) is tuple:
            n=len(gene)
            nrow = int(np.sqrt(n))
            ncol = int(np.ceil(n / nrow))
            
            if (n % 2 != 0 and n > 3) or nrow * ncol < n:
                ncol = ncol+1
            
            if n<4:
                fig, axs = plt.subplots(nrow, ncol, dpi=100,
                                        figsize=(ncol*size*1.5,
                                                 nrow*size*1.5)
                                        )
            else:
                fig, axs = plt.subplots(nrow, ncol, dpi=100,
                                        figsize=(ncol*size,
                                                 nrow*size)
                                        )

            if nrow*ncol>n:
                for i in range(ncol*nrow - n):
                    fig.delaxes(axs[-1][-(i+1)])      
            if type(axs) != np.ndarray:
                axs = [axs]
            else:
                axs = axs.ravel()
            for i in range(n):
                if i < n:
                    if type(gene[i]) is list:
                        marker = gene[i]
                    else:
                        marker = [gene[i]]
                        
                    if marker[0]=='library':
                        color=(self.X3.T>0).sum()
                    else:
                        color=self.X3[marker].T.mean()

                    self.style()
                    g = axs[i].scatter(self.embedding[:, 0],
                                       self.embedding[:, 1],
                                       s=points,
                                       c=color,
                                       alpha=1,
                                       cmap='coolwarm'
                                       )
                    axs[i].set_xticks([])
                    axs[i].set_yticks([])
                    axs[i].autoscale(enable=True, axis='both',tight=tight)

                    divider = make_axes_locatable(axs[i])
                    cax = divider.append_axes("right", "2.5%", pad="1%")

                    if marker[0] == 'library':
                        fig.colorbar(g, cax=cax, 
                                        label='library complexity')
                    else:
                        fig.colorbar(g, cax=cax)

                    
                    if title:
                        axs[i].set_title(title)
                    else:
                        if len(marker) < 2:
                            axs[i].set_title(str(marker[0]),fontsize=fontsize-2)
                        elif len(marker) > 1:
                                axs[i].set_title('list starting with ' + str(marker[0]),fontsize=fontsize-2)

                    if i % ncol == 0:
                        axs[i].set_ylabel(xytitle+'2', fontsize=fontsize)
                    if ((i // ncol) + 1) == nrow:
                        axs[i].set_xlabel(xytitle+'1', fontsize=fontsize)

        else:
            with sns.plotting_context("paper", font_scale=1.5):
                self.style()
                g=sns.lmplot(x='x',
                           y='y',
                           fit_reg=False,
                           scatter_kws={'s': points,
                                        'alpha': .9,                                                                                                                   'color':'black'},
                           hue=None,
                           data=pd.DataFrame(self.embedding,
                                             columns=['x', 'y']),
                           height=size,
                           aspect=1,
                           legend=False,
                           )
                g.set(yticks=[],xticks=[])
                plt.xlabel(xytitle + '1', fontsize=fontsize)
                plt.ylabel(xytitle + '2', fontsize=fontsize)
                plt.autoscale(enable=True, axis='both', tight=tight)
                
        sns.despine(top=False, right=False, left=False, bottom=False)

        if title:
            plt.title(title)
        if path:
            plt.savefig(path, bbox_inches='tight')
        plt.show()
        
    def _get_cluster_classes(self, den, label='ivl'):
        cluster_idxs = defaultdict(list)
        for c, pi in zip(den['color_list'], den['icoord']):
            for leg in pi[1:3]:
                i = (leg - 5.0) / 10.0
                if abs(i - int(i)) < 1e-5:
                    cluster_idxs[c].append(int(i))

        cluster_classes = {}
        for c, l in cluster_idxs.items():
            i_l = [den[label][i] for i in l]
            cluster_classes[c] = i_l

        return cluster_classes
  
    def visual_hierarchy(self,
                    thrs=None,
                    path=False,
                    palette=pallete_50,
                    cell_label=False,
                    value_range=5):
        #thrs is te threshold in dendrogram
        # cell_label if one wants to add extra info in columns colors
        
        self.style_mp_stat()
        
        pop = pd.DataFrame(self.X_vis,index=self.normal_cells,columns=self.selected_genes)
        pop=pop.T
        palette_clust=palette

        Y = self.h2
        X = self.h1

        if thrs is None:
            thrs = 0.5*max(Y[:,2])
        else:
            thrs=thrs
        hierarchy.set_link_color_palette(palette_clust)
        denC = hierarchy.dendrogram(Y,labels = pop.columns,color_threshold=thrs,no_plot=True)
        clustersC = self._get_cluster_classes(denC)
        clusterC = []

        for i in pop.columns:
            included=False
            for j in clustersC.keys():
                if i in clustersC[j]:
                    clusterC.append(j)
                    included=True
            if not included:
                clusterC.append(None)

        if cell_label:    
            gC = sns.clustermap(pop,
                    linewidths=0, xticklabels=False, yticklabels=False,cbar_kws={"orientation": "vertical"},
                     center=0, figsize=(8, 6),vmin=-value_range,vmax=value_range
                                ,col_colors=[clusterC,cell_label],col_linkage=Y,row_linkage=X, cmap='coolwarm')
            gC.cax.set_visible(True)
        else:
            gC = sns.clustermap(pop,
                linewidths=0, xticklabels=False, yticklabels=False,vmin=-value_range,vmax=value_range,
                cbar_kws={"orientation": "vertical"}, center=0, figsize=(7, 7),col_colors=clusterC
                                ,col_linkage=Y,row_linkage=X, cmap='coolwarm' )


        if path:
            plt.savefig(path, bbox_inches='tight')        

        tabC=pd.DataFrame()
        for i in set(denC['color_list']): 
            color_cluster=pd.Series(clusterC).apply(lambda x: x==str(i)).astype(int).values.tolist()
            tabC[i]=color_cluster

        if len(tabC.index)==len(pop.columns):
            tabC.index=pop.columns
            tabC=tabC[list(set(tabC.columns.tolist()) & set(palette_clust))]
            plt.show()
            
            for i in range(0,tabC.shape[1]):
                tabC.T.iloc[i]=tabC.T.iloc[i]*(i+1)
                
            excase=tabC.loc[tabC.T.sum()<1].index.tolist()
            if len(excase)>0:
                for i in excase:
                    self.embedding=np.delete(self.embedding,pd.index(excase).get_loc(i),axis=0)
                tabC=tabC.copy()
                tabC.drop(excase,axis=0,inplace=True)

            self.labels_hierarchy=tabC.T.sum().tolist()
            self.palette_hierarchy = tabC.columns.tolist()
     
        else:
            print('threshold too small')
            
    def get_cluster_info(self,
                        labels=False,
                        cluster=False,
                        genes=10,
                        plot_genes=True):
        
        self.style()          
        cl = pd.DataFrame({'cluster': labels},index=self.normal_cells)
        if cluster is 'all':
            return cl
        
        elif type(cluster) is int:
            cl_n = cl.loc[cl['cluster']==cluster].index.tolist()
            cl[cl['cluster']!=cluster]=200000
            cl[cl['cluster']==cluster]=200001
            
            print('The cluster', cluster,'has',len(cl_n),'cells')
            self.plot(labels=cl['cluster'].tolist(),palette=['#d8dcd6','#49759c'])
                    
            print('The top',genes,'highly expressed signal-like genes in this cluster are:')
            top=self.X3.loc[self.normal_cells,self.normal_genes].loc[cl_n,:].mean()\
                    .sort_values(ascending=False)[:genes].index.tolist()
            
            for i in top:
                    print(i)
            
            if plot_genes:
                self.plot(gene=tuple(top))
        
        else:
            print('Select right cluster option')
          
      
    def get_gene_info(self,
                      labels=False,
                      gene=False,
                     path=False,
                     legend=False,
                     size=6.5):
        
        self.style()
        if 0 in labels or 200001 in labels:
            palette=pallete_50
        else:
            palette=self.palette_hierarchy

        cl=pd.DataFrame({'cluster': labels},index=self.normal_cells)
        cl2=[self.X3[gene].T.mean()[cl.loc[cl['cluster']==i].index] for i in pd.unique(labels)]

        if type(legend) is list:
            order=legend
            index_lab=legend
        else:
            order=sorted(pd.unique(labels))
            index_lab=pd.unique(labels)

        cl2=pd.DataFrame(cl2,index=index_lab)
        
        fig = plt.figure(figsize=[size, 5])       
        sns.violinplot(data=cl2.T,orient='v', inner=None,order=order,
                      palette=palette)     
        sns.stripplot(data=cl2.T,color="black",size=2.5,order=order)
        
        if legend:
            plt.xticks(rotation=45)

        plt.xlabel('Cluster')
        plt.ylabel('log2(1+TPM)')
                      
        if path:
            plt.savefig(os.path.splitext(path)[0]+'_violin'+os.path.splitext(path)[1]
                        , bbox_inches='tight') 
        
        #------Fig2 ------------
        
        cl['gene']=self.X3[gene].T.mean()        
        palette2=[]
        pal_list=cl.copy()
        for i in sorted(pd.unique(cl['cluster'])):
            if cl[cl['cluster']==i]['gene'].var()<0.0000000001:
                cl.drop(cl[cl['cluster']==i].index,axis=0,inplace=True)
            else:
                palette2.append(palette[sorted(pd.unique(pal_list['cluster'])).index(i)])
        if len(palette2)>0:
            palette=palette2
            
        sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
              
        g = sns.FacetGrid(cl, row="cluster", hue="cluster", aspect=8.5,
                          height=.7,palette=palette)
        g.map(sns.kdeplot, "gene", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.13)
        g.map(plt.axhline, y=0, lw=2, clip_on=False)

        bottom, top = plt.xlim()
        down, up = plt.ylim()
        for i in g.row_names:
            if type(legend) is list:
                mi=legend[g.row_names.index(i)]
                g.axes[g.row_names.index(i),0].text(bottom-len(mi)*0.35,up*0.04,mi,
                               color=palette[g.row_names.index(i)],fontsize='medium')        
            else:
                mi=i        
                g.axes[g.row_names.index(i),0].text(bottom-len('cluster ')*0.35,up*0.04,'cluster '+str(mi),
                               color=palette[g.row_names.index(i)],fontsize='medium')

        if len(palette)<11 and len(palette)>5:
            g.fig.subplots_adjust(hspace=-0.7,top = 0.70,bottom = 0.05 )
        elif len(palette)<6:
            g.fig.subplots_adjust(hspace=-0.6,top = 0.85,bottom = 0.05 )
        else:
            g.fig.subplots_adjust(hspace=-0.7,top = 0.50,bottom = 0.05 )

        g.set_titles("")
        g.set(yticks=[])
        g.despine(bottom=True, left=True)
        plt.xlabel('log2(1+TPM)')
        
        if path:
            plt.savefig(os.path.splitext(path)[0]+'_ridge'+os.path.splitext(path)[1],
                        bbox_inches='tight') 
        self.style()
        plt.show()