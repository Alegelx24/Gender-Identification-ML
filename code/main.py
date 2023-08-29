
import dataset_util as util
import validate as validate
import gaussian_classifier as gaussian
import logistic_regression as logreg
import support_vector_machine as svm
import gaussian_mixture_model as gmm

k=5 #K-fold

  
def main():
    D,L = util.load_training_set('./data/Train.txt')
    D_norm= util.scale_ZNormalization(D)[0]
    #D_gaussianized= util.gaussianize_training(D)
    
    
    gaussianize= False 

    #plot(D, L, gaussianize) #plot raw features before gaussianization
    #gaussianize= True
    #plot(D_gaussianized, L, gaussianize) #plot gaussianized features   

    #TRAINING
    print("VALIDATION WITHOUT GAUSSIANIZATION")

    #train_evaluate_gaussian_models(D, L)
    #train_evaluate_gaussian_models_zscore(D_norm, L)
   
    #logreg.plot_minDCF_wrt_lamda(D_norm, L, gaussianize=False)
    #logreg.quadratic_plot_minDCF_wrt_lambda(D_norm, L, gaussianize=False)
    #train_evaluate_logreg(D, L)
    
    
    #svm.plot_linear_minDCF_wrt_C(D, L, gaussianize)
    #svm.plot_quadratic_minDCF_wrt_C(D, L, gaussianize)
    svm.plot_RBF_minDCF_wrt_C(D, L, gaussianize)
    #train_evaluate_svm(D,L)

    #train_evaluate_gmm(D, L)
    
    



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
    while m>=9:
        
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
                print(" Logistic regression -piT = %f - pi = %f  minDCF = %f" %(Options['piT'], pi,min_dcf_kfold))
             
        if m < 12:
            D = util.pca(m, D)[0]
            print ("################################################")
            print ("### Quadratic Logistic regression with m = %d ##" %m)
            print ("###############################################")
        else:
            print ("###################################################")
            print ("#### Quadratic Logistic regression with NO PCA ####")
            print ("##################################################")
            
        for piT in [0.1, 0.5, 0.9]:
            Options['lambdaa']=100
            Options['piT']=piT
            for pi in [0.1, 0.5, 0.9]:
                min_dcf_kfold = validate.kfold(D, L, k, pi, logreg.compute_score_quadratic, Options)[0]
                print(" Quadratic Log Reg -piT = %f - pi = %f  minDCF = %f" %(Options['piT'], pi,min_dcf_kfold))  
        
        m = m-1
        D=D_copy #restore original dataset
    


def train_evaluate_svm(D,L):
    D_copy = D
    Options={
        'C' : None,
        'piT': None,
        'gamma':None,
        'rebalance':None
        }  
    m = 12
    while m>=9:
        
        if m < 12:
            D = util.pca(m, D)[0]
            print ("##########################################")
            print ("############ SVM LINEAR with m = %d #######" %m)
            print ("##########################################")
        else:
            print ("##########################################")
            print ("######## SVM LINEAR with NO PCA ##########")
            print ("##########################################")
            
        Options['C']=0.1
        for piT in [0.1, 0.5, 0.9]:
            for pi in [0.1, 0.5, 0.9]:
                Options['piT']=piT
                Options['rebalance']=True
                min_dcf_kfold = validate.kfold(D, L, k, pi, svm.compute_score_linear, Options)[0]
                print("Linear SVM -piT = %f -C=%f - pi = %f - minDCF = %f" %(piT,Options['C'], pi,min_dcf_kfold))
                
        Options['rebalance']=False
        for pi in [0.1, 0.5, 0.9]:
                min_dcf_kfold = validate.kfold(D, L, k, pi, svm.compute_score_linear, Options)[0]
                print("Linear SVM without rebalancing -C=%f - pi = %f - minDCF = %f" %(Options['C'], pi,min_dcf_kfold)) 
          
        if m < 12:
            D = util.pca(m, D)[0]
            print ("##########################################")
            print ("########## SVM QUADRATIC with m = %d ######" %m)
            print ("##########################################")
        else:
            print ("##########################################")
            print ("########SVM QUADRATIC with NO PCA ########")
            print ("##########################################")
        for piT in [0.5]:
            for pi in [0.5, 0.9]:
                Options['C']=1e-03
                Options['piT']=piT
                Options['rebalance']=True
                min_dcf_kfold = validate.kfold(D, L, k, pi, svm.compute_score_quadratic, Options)[0]
                print("Quadratric SVM -piT = %f -C=%f - pi = %f - minDCF = %f" %(piT,Options['C'], pi,min_dcf_kfold))
                
        Options['rebalance']=False
        Options['C']=0.1##################################################################################################
        for pi in [0.1]:
            min_dcf_kfold = validate.kfold(D, L, k, pi, svm.compute_score_quadratic, Options)[0]
            print("Quadratic SVM without rebalancing -C=%f - pi = %f - minDCF = %f" %(Options['C'], pi,min_dcf_kfold))
           
        if m < 12:
            D = util.pca(m, D)[0]
            print ("##########################################")
            print ("######### SVM RBF with m = %d #############" %m)
            print ("##########################################")
        else:
            print ("##########################################")
            print ("##########SVM RBF with NO PCA ############")
            print ("##########################################")
        for piT in [0.5]:
            for pi in [0.9]:
                Options['C']=10###############################################################################################
                Options['piT']=piT
                Options['gamma']=0.01
                Options['rebalance']=True
                min_dcf_kfold = validate.kfold(D, L, k, pi, svm.compute_score_RBF, Options)[0]
                print("RBF SVM -piT = %f -gamma =%f -C=%f - pi = %f -> minDCF = %f" %(piT, Options['gamma'], Options['C'], pi,min_dcf_kfold))      
            
        Options['rebalance']=False
        for pi in [0.1, 0.5, 0.9]:
            pass
            min_dcf_kfold = validate.kfold(D, L, k, pi, svm.compute_score_RBF, Options)[0]
            print("RBF SVM without rebalancing -gamma =%f -C=%f - pi = %f -> minDCF = %f" %(Options['gamma'], Options['C'], pi,min_dcf_kfold))
              
        m = m-1
        D=D_copy #restore original dataset


        
def train_evaluate_gmm(D,L):
    D_copy = D
    Options={ 
        'Type':None,
        'iterations':None #gmm components = 2^iterations
        }  
    m = 12
    while m>=9:
        if m < 12:
            D = util.pca(m, D)[0]
            print ("##########################################")
            print ("############# GMM with m = %d ##############" %m)
            print ("##########################################")
        else:
            print ("##########################################")
            print ("########## GMM LINEAR with NO PCA ########")
            print ("##########################################")
            
            Options['Type']='full'
            Options['iterations']= 4
            _, scores_full, labels = validate.kfold(D, L, k, 0.5, gmm.compute_score, Options) #pi=0.5 actually not used to compute scores
           
            Options['Type']='diag'
            Options['iterations']= 5
            _, scores_diag, labels = validate.kfold(D, L, k, 0.5, gmm.compute_score, Options) #pi=0.5 actually not used to compute scores
            
            Options['Type']='full-tied'
            Options['iterations']= 6
            _, scores_full_tied, labels = validate.kfold(D, L, k, 0.5, gmm.compute_score, Options) #pi=0.5 actually not used to compute scores
            
            Options['Type']='diag-tied'
            Options['iterations']= 6
            _, scores_diag_tied, labels = validate.kfold(D, L, k, 0.5, gmm.compute_score, Options) #pi=0.5 actually not used to compute scores

        for pi in [0.1, 0.5, 0.9]:

            Options['Type']='full'
            Options['iterations']= 4
            min_dcf_kfold = validate.compute_min_DCF(scores_full, labels, pi, 1, 1)
            print(" gmm %s -components=%d - pi = %f --> minDCF = %f" %(Options['Type'], 2**Options['iterations'], pi,min_dcf_kfold))
            
            Options['Type']='diag'
            Options['iterations']= 5
            min_dcf_kfold = validate.compute_min_DCF(scores_diag, labels, pi, 1, 1)
            print(" gmm %s -components=%d - pi = %f --> minDCF = %f" %(Options['Type'], 2**Options['iterations'], pi,min_dcf_kfold))
            
            Options['Type']='full-tied'
            Options['iterations']= 6
            min_dcf_kfold = validate.compute_min_DCF(scores_full_tied, labels, pi, 1, 1)
            print(" gmm %s -components=%d - pi = %f --> minDCF = %f" %(Options['Type'], 2**Options['iterations'], pi,min_dcf_kfold))
            
            Options['Type']='diag-tied'
            Options['iterations']= 6
            min_dcf_kfold = validate.compute_min_DCF(scores_diag_tied, labels, pi, 1, 1)
            print(" gmm %s -components=%d - pi = %f --> minDCF = %f" %(Options['Type'], 2**Options['iterations'], pi,min_dcf_kfold))
                
        m = m-1
        D=D_copy #restore original dataset
            
   
        


    
main()