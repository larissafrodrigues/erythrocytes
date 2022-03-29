'''
Morphological analysis and classification of erythrocytes in microscopy images 
Larissa F. Rodrigues
2016
'''

import numpy as np
from sklearn import cross_validation, feature_selection
from sklearn import tree,neighbors, preprocessing, metrics
import math, os, copy
from scipy import misc
import scipy.ndimage as ndi
from skimage import util, color, feature, measure, exposure, morphology, filters, segmentation
from sklearn.utils import shuffle
from sklearn.metrics.classification import accuracy_score
import matplotlib.pyplot as plt

##############################################################################
# Initial parameters
NUM_ROUNDS = 10;
num_folds  = 10;
SIZE_FEATS   = 9;  # Numero de caracteristicas
#classif    = 'tree'
classif    = 'knn'

# Contagem automatica do numero de experimentos.
file_ = open('counter.txt','r+')
exp_count = int(file_.read())
print 'Contador de experimentos: ', exp_count, type(exp_count)
str_count = str(exp_count+1)
file_.seek(0)
file_.write(str_count)
file_.close()

exp_dirname = 'exp_' + str(exp_count) + '_'  + classif 

if not os.path.exists(exp_dirname):
    os.mkdir(exp_dirname)
    os.mkdir(exp_dirname + '\im_bw_median')
    os.mkdir(exp_dirname + '\im_otsu')
    os.mkdir(exp_dirname + '\im_clear_border')
    os.mkdir(exp_dirname + '\im_dilatacao')
    os.mkdir(exp_dirname + '\im_fill_holles')
    os.mkdir(exp_dirname + '\im_erosao')
    os.mkdir(exp_dirname + '\im_seg')
    os.mkdir(exp_dirname + '\im_contour')
    os.mkdir(exp_dirname + '\im_mask')

# ============================================================================
def draw_contour(im, im_bw):
    ###
    # Binary morphological gradient
    im_bw_c = morphology.binary_dilation(im_bw) - im_bw
    ### print im_bw_c.max(), im_bw_c.min()
    
    # Individual channels
    im_R = copy.deepcopy(im)
    im_G = copy.deepcopy(im)
    im_B = copy.deepcopy(im)
    # Contorno vermelho
    im_R[im_bw_c==True] = 1.0
    im_G[im_bw_c==True] = 0.0
    im_B[im_bw_c==True] = 0.0
    
    # Initialize RGB image
    im_ = np.zeros([im_bw.shape[0], im_bw.shape[1], 3], dtype='float')
    
    im_[:,:,0] = im_R[:,:] # RED
    im_[:,:,1] = im_G[:,:] # GREEN
    im_[:,:,2] = im_B[:,:] # BLUE
    ### print type(im_), im_.dtype, im_.shape, im_.max(), im_.min()
    
    return im_
# -----------------------------------------------------------------------------
    
# =============================================================================
def otsu(im):
    ################
    # Median filter.
    im = ndi.median_filter(im_, 3)
    name_out = exp_dirname + '\\im_bw_median\\' + im_nomes[:-4] + '.tif'
    misc.imsave(name_out, im)  
    ##################################
    # Segmentacao pelo método de Otdu
    otsu_th = filters.threshold_otsu(im)
    im_bw = im <= otsu_th

    name_out = exp_dirname + '\\im_otsu\\' + im_nomes[:-4] + '.tif'
    misc.imsave(name_out, im_bw)    

    ##########################################
    # Remove componentes conectados as bordas    
    im_bw = segmentation.clear_border(im_bw)
    name_out = exp_dirname + '\\im_clear_border\\' + im_nomes[:-4] + '.tif'
    misc.imsave(name_out, im_bw)   
    
    ###################################
    # Matriz de rotulos
    # im_L = measure.label(im_r)       # skimage
    [im_L, im_L_size] = ndi.label(im_bw)  # ndimage///
    # Seleciona apenas o maior objeto.
    if im_L.max()>1: # Maior valor presente na matriz de rotulos
        max_area = -np.Inf
        max_area_i = -1
        for i in range(1, im_L.max()+1):
            if np.count_nonzero(im_L==i) > max_area:
                max_area = np.count_nonzero(im_L==i)
                max_area_i = i
         
        im_bw = np.zeros(im_L.shape);
        im_bw[im_L==max_area_i] = 1.;
        # Matriz de rotulos
        # im_L = measure.label(im_tmp)
        [im_L, im_L_size] = ndi.label(im_bw)
    
    ########################
    # Elemento estruturante.
    #### ee = np.array([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])
    ee = morphology.disk(4)
    
    ##########
    #Dilatacao
    im_bw = morphology.binary_dilation(im_bw, selem=ee);
    name_out = exp_dirname + '\\im_dilatacao\\' + im_nomes[:-4] + '.tif'
    misc.imsave(name_out, im_bw)   
    #############
    # Fill holes.
    im_bw = ndi.binary_fill_holes(im_bw)
    name_out = exp_dirname + '\\im_fill_holles\\' + im_nomes[:-4] + '.tif'
    misc.imsave(name_out, im_bw) 
    #######
    #Erosao
    im_bw = morphology.binary_erosion(im_bw, selem=ee)
    name_out = exp_dirname + '\\im_erosao\\' + im_nomes[:-4] + '.tif'
    misc.imsave(name_out, im_bw) 

    return im_bw

##########################################################
# Lista contendo os nomes dos arquivos no diretorio 'path'
path = 'img/'
lista = os.listdir(path)   
# Numero de imagens.
num_imgs = len(lista) 

############################
# Matriz de caracteristicas.
F = np.array([])        # Matriz de Carateristicas
T = np.arange(num_imgs) # Vetor de alvos

# Lista de caracteristicas.
caracteristicas = np.array([])

# Contador de imagens
im_cont = 0

###############################################################################
# IMAGE SEGMENTATION
###############################################################################

#########################################
# Percorre todas as imagens no diretorio.
for im_nomes in lista:
    # Abre a imagem
    im = misc.imread(path + im_nomes)
    # Converte imagem RGB para niveis de cinza
    im_ = color.rgb2gray(im)
    # Converte para float (0,1)
    im_ = util.img_as_float(im_)
    
    #############################
    # Segmentação - Método SCOTTI
    ### im_seg = seg_scotti_1(im_, 3)
    ### im_seg = seg_scotti_2(im_, 3)
    im_seg = otsu(im_)
    
    ###
    # Grava as imagens em disco
    [im_L, im_L_size] = ndi.label(im_seg)

    # GRAVA A IMAGEM SEGMENTADA
    name_out = exp_dirname + '\\im_seg\\' + im_nomes[:-4] + '.tif'
    misc.imsave(name_out, im_seg)     
    # GRAVA A IMAGEM FILTRADA MASCARADA PELA IMG SEGMENTADA
    name_out = exp_dirname + '\\im_mask\\' + '' + im_nomes[:-4] + '.tif'
    misc.imsave(name_out, im_seg*im_)
    # GRAVA A IMAGEM FILTRADA MASCARADA PELA IMG SEGMENTADA
    name_out = exp_dirname + '\\im_contour\\' + '' + im_nomes[:-4] + '.tif'
    misc.imsave(name_out, draw_contour(im_, im_seg))
    
    ###########################################################################
    # EXTRACAO DE CARACTERISTICAS
    # Obtem propriedades (caracteriticas) de regioes rotuladas em uma imagem.
    ###########################################################################
    props = measure.regionprops(im_L, im_)
    
    F_i  = np.array([props[0].area,            
                     props[0].convex_area,     
                     props[0].eccentricity,    
                     props[0].equivalent_diameter,  
                     props[0].extent, 
                     props[0].perimeter, 
                     props[0].solidity, 
                     4*math.pi*props[0].area/props[0].perimeter*props[0].perimeter, # circularity
                     props[0].mean_intensity 
                     ])
    
    ###########################
    # Build the FEATURE MATRIX.
    if im_cont == 0:
        F = F_i
    else:
        F = np.vstack((F,F_i))

    # Increase image counter
    im_cont = im_cont + 1
    
# Setting Targets    
for i in range(1, num_imgs):
    if (i >= 1 and i <=201):
        T[i] = 0
    if (i >= 202 and i <=412):
        T[i] = 1    
    if (i >=413 and i <=626):
        T[i] = 2    
        
# Univariate feature selection using ANOVA
F_indices = np.arange(F.shape[-1])

b = feature_selection.SelectKBest(feature_selection.f_classif, k=3)
### b = feature_selection.SelectKBest(feature_selection.chi2, k=3)
F_new = b.fit_transform(F, T)
print '---- ANOVA based feature selection ----'
# print F.shape
# print F_new
# print F_new.shape
# print b.get_support()
scores = -np.log10(b.pvalues_)
print 'P-values A: ', b.pvalues_
print 'P-values B: ', np.around(b.pvalues_, 32)
scores /= scores.max()
print 'Scores: ', np.around(scores, 4)

# Ordena scores!!!
scores_ = np.argsort(scores)
scores_ = scores_[::-1]

print 'Ordened scores: ', np.around(scores, 4)
print 'Ordened scores indexes: ', scores_

# FIGURE
plt.figure(1)
plt.bar(F_indices - .45, scores, width=.2,
        label=r'Univariate score ($-Log(p_{value})$)', color='g')
plt.title("Comparing feature selection")
plt.xlabel('Feature number')
### plt.yticks(())
plt.axis('tight')
plt.legend(loc='upper right')
# FIGURE
plt.figure(2)
plt.bar(F_indices - .45, scores[scores_], width=.2,
        label=r'Univariate score ($-Log(p_{value})$)', color='g')
plt.title("Comparing feature selection")
plt.xlabel('Feature number')
plt.yticks(())
plt.axis('tight')
plt.legend(loc='upper right')

#### plt.show()
print '---------------------------------------'

###############################################################################
# CLASSIFIER SELECTION
if classif=='tree':
    clf = tree.DecisionTreeClassifier()
elif classif=='knn':
    clf = neighbors.KNeighborsClassifier(n_neighbors=7)

# Create file with filename
filename_out1 = exp_dirname + '\\VALIDATION_' + classif + ".csv"
filename_out2 = exp_dirname + '\\SUMMARY_' + classif + ".csv"
filename_out3 = exp_dirname + '\\INDIVIDUAL_' + classif + ".csv"

file1 = open(filename_out1,'w')
file2 = open(filename_out2,'w')
file3 = open(filename_out3,'w')

########################################
# Testa subconjuntos de caracteristicas
for num_feat in range(1,SIZE_FEATS+1): # num_features
    print 'Number of features: ', num_feat

    #Select the 'num_feat' best features
    F_s = F[:,scores_[0:num_feat]]    

    ############################################################################
    # CLASSIFICATION
    ############################################################################
    
    #########################################
    # Define o gerador de numeros aleatorios.
    random_state = np.random.RandomState(0)

    ###########################################
    # Media dos indices para n caracteristicas.
    feat_p_mean = feat_r_mean = feat_f_mean = feat_s_mean = feat_a_mean = 0.0
    
    # Considera a media do todas as 3 classes.
    feat_class = np.zeros(5)
    print feat_class
    
    # Repete o k-fold 10 vezes.
    for round in range(1,NUM_ROUNDS+1):
        ### print '\n=======================================\nRound: ', round
        # Embaralha a matriz de caracteristicas.
        F_, T_ = shuffle(F_s, T, random_state=random_state)
        ####kf = cross_validation.StratifiedKFold(T, n_folds=10)
        kf = cross_validation.KFold(num_imgs, n_folds=num_folds)
        
        ##########################################
        # Media de todos os FOLDS para este ROUND.
        round_p_mean = round_r_mean = round_f_mean = round_s_mean = round_a_mean = 0.0
        ##
        #
        round_class = np.zeros(5)
        print round_class
        
        # Fold counters
        fold=0
        for train, test in kf:
            print 'Features %d; Round %d; Fold %d' % (num_feat, round, fold)
            
            # Separate training and test subsets
            X_train, X_test, y_train, y_test = F_[train], F_[test], T_[train], T_[test]
            
            # Transforma os vetores de caracteristicas
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train_transformed = scaler.transform(X_train)
            X_test_transformed = scaler.transform(X_test)
            # ---- TEST ----
            #print 'X_train_transformed: ', X_train_transformed.mean(), X_train_transformed.std()
            #print 'X_test_transformed: ', X_test_transformed.mean(), X_test_transformed.std()
            
            # Treina o classificador
            clf.fit(X_train_transformed, y_train)
            # Acuracia da classificacao    
            clf.score(X_test_transformed, y_test)  
            # Predicao
            classificacao = clf.predict(X_test_transformed)
            # Classification report
            #### print metrics.confusion_matrix(y_test, classificacao)
            #print metrics.classification_report(y_test, classificacao)
            [p, r, f, s] = metrics.precision_recall_fscore_support(y_test, classificacao)
            a = metrics.accuracy_score(y_test, classificacao)

            # Save RESULTS - Detailed.
            str_a1 = "FEATURES, %d, Accuracy  , %.4f \n" % (num_feat, a)
            file3.write(str_a1)

            #######################################
            # Soma dos indices para todos os FOLDS.
            round_p_mean = round_p_mean + p
            round_r_mean = round_r_mean + r 
            round_f_mean = round_f_mean + f
            round_s_mean = round_s_mean + s
            round_a_mean = round_a_mean + a
            
            # #####################
            # Mean for all classes.
            round_class_tmp = [ p.mean(), r.mean(), f.mean(), s.sum(), a ]
            round_class = round_class + round_class_tmp
            ##############################
            # Incrementa FOLD.
            fold += 1
          
        ##################################  
        # Compute the mean of the indexes.   
        round_p_mean = round_p_mean / num_folds 
        round_r_mean = round_r_mean / num_folds
        round_f_mean = round_f_mean / num_folds
        round_s_mean = round_s_mean / num_folds
        round_a_mean = round_a_mean / num_folds
        
# #################################################  
# # Compute the mean of the indexes for each class.   
# round_p_mean_cl = round_p_mean.mean() 
# round_r_mean_cl = round_r_mean.mean()
# round_f_mean_cl = round_f_mean.mean()
# round_s_mean_cl = round_s_mean # The support is summed-up
# round_a_mean_cl = round_a_mean.mean()
        
        # #####################
        # Mean for all classes.
        round_class = round_class / num_folds

#         print '\n---- Medias - Folds ----'
#         print 'Precision: ', media_p
#         print 'Recall:    ', media_r
#         print 'F-Score:   ', media_f
#         print 'Support:   ', media_s
#         print 'Acuracia:  ', media_a

        ###################################
        # Soma dos indices de cada feature.
        feat_p_mean = feat_p_mean + round_p_mean
        feat_r_mean = feat_r_mean + round_r_mean
        feat_f_mean = feat_f_mean + round_f_mean
        feat_s_mean = feat_s_mean + round_s_mean
        feat_a_mean = feat_a_mean + round_a_mean
        ##############
        #
        feat_class = feat_class + round_class
        
        #################
        # Increase round.
        round += 1
    
    #######################################
    # Medias dos indices para cada feature.   
    feat_p_mean = feat_p_mean / NUM_ROUNDS
    feat_r_mean = feat_r_mean / NUM_ROUNDS
    feat_f_mean = feat_f_mean / NUM_ROUNDS
    feat_s_mean = feat_s_mean / NUM_ROUNDS
    feat_a_mean = feat_a_mean / NUM_ROUNDS
    #############
    #
    feat_class = feat_class / NUM_ROUNDS
    
    ##########################
    # Save RESULTS - Detailed.
    str_c = "FEATURES , %d , %s \n" % (num_feat, np.array_str(scores_[0:num_feat]) )
    str_p = "Precision , %.4f , %.4f , %.4f \n" % ( feat_p_mean[0], feat_p_mean[1], feat_p_mean[2] )
    str_r = "Recall    , %.4f , %.4f , %.4f \n" % ( feat_r_mean[0], feat_r_mean[1], feat_r_mean[2] )
    str_f = "F-Score   , %.4f , %.4f , %.4f \n" % ( feat_f_mean[0], feat_f_mean[1], feat_f_mean[2] )
    str_s = "Support   , %.4f , %.4f , %.4f \n" % ( feat_s_mean[0], feat_s_mean[1], feat_s_mean[2] )
    str_a = "Accuracy  , %.4f \n" % feat_a_mean
    
    file1.write(str_c)
    file1.write(str_p)
    file1.write(str_r)
    file1.write(str_f)
    file1.write(str_s)
    file1.write(str_a)
    
    ############################
    # Save RESULTS - Summarized.
    #### str_summary = "%d , %.6f , %.6f , %.6f , %.6f , %.6f \n" % ( num_feat, feat_p_mean.mean(), feat_r_mean.mean(), feat_f_mean.mean(), feat_s_mean.mean(), feat_a_mean.mean())
    #### file2.write(str_summary)
    str_summary2 = "%d , %.6f , %.6f , %.6f , %.6f , %.6f \n" % ( num_feat, feat_class[0], feat_class[1], feat_class[2], feat_class[3], feat_class[4] )
    file2.write(str_summary2)
    
##################
# Fecha o arquivo.
file1.close()
file2.close()
file3.close()
plt.show()   