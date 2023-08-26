
import dataset_util as util
import validate as validate
import gaussian_classifier as gaussian
import logistic_regression as logreg

k=5 #K-fold

  
def main():
    D,L = util.load_training_set('./data/Train.txt')
    D_norm= util.scale_ZNormalization(D)[0]
    #D_gaussianized= util.gaussianize_training(D)
    
    gaussianize= False 

    #plot(D, L, gaussianize) #plot raw features before gaussianization
    gaussianize= True
    #plot(D_gaussianized, L, gaussianize) #plot gaussianized features   

    #TRAINING
    #print("VALIDATION WITHOUT GAUSSIANIZATION")

    #train_evaluate_gaussian_models(D, L)
    #train_evaluate_gaussian_models_zscore(D_norm, L)
   
    #logreg.plot_minDCF_wrt_lamda(D, L, gaussianize)
    #logreg.quadratic_plot_minDCF_wrt_lambda(D, L, gaussianize)
    train_evaluate_logreg(D, L)
    
    



def plot(DTR, LTR, gaussianize):
    #save histograms of the distribution of all the features in '../Images' folder. E
    util.plot_histograms(DTR, LTR,gaussianize)
    util.plot_scatters(DTR, LTR,gaussianize)
    util.heatmap_generator(DTR, LTR, "correlation_heatmap")




def train_evaluate_gaussian_models(D,L):
    Options={ }  
    D_copy = D
    m = 12
    while m>=7:
        if m < 12:
            D = util.pca(m, D)[0]
            print ("##########################################")
            print ("##### Gaussian classifiers with m = %d ####" %m)
            print ("##########################################")
        else:
            print ("##########################################")
            print ("#### Gaussian classifiers with NO PCA ####")
            print ("##########################################")
                    
        
        for pi in [0.1, 0.5, 0.9]:
            min_dcf_full = validate.kfold(D, L, k, pi, gaussian.compute_score_full, Options)[0]
            print(" Full-Cov - pi = %f -> minDCF = %f" %(pi,min_dcf_full))
            min_dcf_diag = validate.kfold(D, L, k, pi, gaussian.compute_score_diag, Options)[0]
            print(" Diag-cov - pi = %f -> minDCF = %f" %(pi,min_dcf_diag))
            min_dcf_tied_full = validate.kfold(D, L, k, pi, gaussian.compute_score_tied_full, Options)[0]
            print(" Tied full-cov - pi = %f  minDCF = %f" %(pi,min_dcf_tied_full))
            min_dcf_tied_diag = validate.kfold(D, L, k, pi, gaussian.compute_score_tied_diag, Options)[0]
            print(" Tied diag-cov - pi = %f  minDCF = %f" %(pi,min_dcf_tied_diag))

        m=m-1
        D=D_copy #restore original dataset



def train_evaluate_gaussian_models_zscore(D,L):
    Options={ }  
    D_copy = D
    m = 12
    while m>=7:
        if m < 12:
            D = util.pca(m, D)[0]
            print ("##########################################")
            print ("##### Gaussian classifiers with m = %d and Z-score norm ####" %m)
            print ("##########################################")
        else:
            print ("##########################################")
            print ("#### Gaussian classifiers with NO PCA and Z-score norm ####")
            print ("##########################################")
                    
        
        for pi in [0.1, 0.5, 0.9]:
            min_dcf_full = validate.kfold(D, L, k, pi, gaussian.compute_score_full, Options)[0]
            print(" Full-Cov - pi = %f -> minDCF = %f" %(pi,min_dcf_full))
            min_dcf_diag = validate.kfold(D, L, k, pi, gaussian.compute_score_diag, Options)[0]
            print(" Diag-cov - pi = %f -> minDCF = %f" %(pi,min_dcf_diag))
            min_dcf_tied_full = validate.kfold(D, L, k, pi, gaussian.compute_score_tied_full, Options)[0]
            print(" Tied full-cov - pi = %f  minDCF = %f" %(pi,min_dcf_tied_full))
            min_dcf_tied_diag = validate.kfold(D, L, k, pi, gaussian.compute_score_tied_diag, Options)[0]
            print(" Tied diag-cov - pi = %f  minDCF = %f" %(pi,min_dcf_tied_diag))

        m=m-1
        D=D_copy #restore original dataset

       

        
def train_evaluate_logreg(D,L):
    D_copy = D
    Options={
    'lambdaa' : None,
    'piT': None,
    }  
    m = 12
    while m>=8:
        
        if m < 12:
            D = util.pca(m, D)[0]
            print ("############################################")
            print ("### Linear Logistic regression with m = %d ##" %m)
            print ("###########################################")
        else:
            print ("################################################")
            print ("#### Linear Logistic regression with NO PCA ####")
            print ("###############################################")
            
        for piT in [0.1, 0.5, 0.9]:
            Options['lambdaa']=1e-06
            Options['piT']=piT
            for pi in [0.1, 0.5, 0.9]:
                min_dcf_kfold = validate.kfold(D, L, k, pi, logreg.compute_score, Options)[0]
                print(" Logistic reggression -piT = %f - pi = %f  minDCF = %f" %(Options['piT'], pi,min_dcf_kfold))
             
        if m < 8:
            D = util.pca(m, D)[0]
            print ("################################################")
            print ("### Quadratic Logistic regression with m = %d ##" %m)
            print ("###############################################")
        else:
            print ("###################################################")
            print ("#### Quadratic Logistic regression with NO PCA ####")
            print ("##################################################")
            
        for piT in [0.1, 0.5, 0.9]:
            Options['lambdaa']=1e-06
            Options['piT']=piT
            for pi in [0.1, 0.5, 0.9]:
                min_dcf_kfold = validate.kfold(D, L, k, pi, logreg.compute_score_quadratic, Options)[0]
                print(" Quadratic Log Reg -piT = %f - pi = %f  minDCF = %f" %(Options['piT'], pi,min_dcf_kfold))  
        
        m = m-1
        D=D_copy #restore original dataset
    



    
main()