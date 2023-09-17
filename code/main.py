
import dataset_util as util
import validate as validate
import gaussian_classifier as gaussian
import logistic_regression as logreg
import support_vector_machine as svm
import gaussian_mixture_model as gmm
import score_calibration as calibration
import model_fusion as fusion
import evaluation as eval
k=5 #K-fold

  
def main():
    D,L = util.load_training_set('./data/Train.txt')
    D_centered=util.center_dataset(D)[0]
    D_norm= util.scale_ZNormalization(D)[0]
    #D_gaussianized= util.gaussianize_training(D)
    
    
    gaussianize= False 

    #plot(D, L, gaussianize) #plot raw features before gaussianization
    #gaussianize= True
    #plot(D_gaussianized, L, gaussianize) #plot gaussianized features   

    #TRAINING
    print("VALIDATION WITHOUT ZSCORE NORMALIZATION")

    #train_evaluate_gaussian_models(D_norm, L)
   
    #logreg.plot_minDCF_wrt_lamda(D, L, gaussianize=False)
    #logreg.quadratic_plot_minDCF_wrt_lambda(D, L, gaussianize=False)
    #train_evaluate_logreg(D_norm, L)
    
    
    #svm.plot_linear_minDCF_wrt_C(D, L, gaussianize)
    #svm.plot_quadratic_minDCF_wrt_C(D, L, gaussianize)
    #svm.plot_RBF_minDCF_wrt_C(D, L, gaussianize)
    #train_evaluate_svm(D,L)

    #gmm.plot_minDCF_wrt_components(D, D_norm, L)
    #train_evaluate_gmm(D_norm, L)
    
    
    #print("VALIDATION WITH ZSCORE NORMALIZATION")
    #train_evaluate_gaussian_models_zscore(D_norm, L)
   
    #logreg.plot_minDCF_wrt_lamda(D_norm, L, gaussianize=False)
    #logreg.quadratic_plot_minDCF_wrt_lambda(D_norm, L, gaussianize=False)
    #train_evaluate_logreg(D_Norm, L)

    
    #svm.plot_linear_minDCF_wrt_C(D_norm, L, gaussianize)
    #svm.plot_quadratic_minDCF_wrt_C(D_norm, L, gaussianize)
    #svm.plot_RBF_minDCF_wrt_C(D_norm, L, gaussianize=True)
    train_evaluate_svm(D_norm,L)

    
    #validate.two_bests_roc(D, L) #model selection
    #perform_calibration(D_norm,L)
    #validate_fusion(D_norm,L)
    #eval.evaluation()

def plot(DTR, LTR, gaussianize):
    #save histograms of the distribution of all the features in '../Images' folder. E
    util.plot_fraction_explained_variance_pca(DTR, LTR)
    #util.plot_histograms(DTR, LTR,gaussianize)
    #util.plot_scatters(DTR, LTR,gaussianize)
    #util.heatmap_generator(DTR, LTR, "correlation_heatmap")




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
            Options['lambdaa']=1e-05    
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
            print ("########## SVM QUADRATIC with m = %d ######" %m)
            print ("##########################################")
        else:
            print ("##########################################")
            print ("########SVM QUADRATIC with NO PCA ########")
            print ("##########################################")
        for piT in [0.5]:
            for pi in [0.5, 0.9]:
                Options['C']=0.1
                Options['piT']=piT
                Options['rebalance']=True
                #min_dcf_kfold = validate.kfold(D, L, k, pi, svm.compute_score_quadratic, Options)[0]
                #print("Quadratric SVM -piT = %f -C=%f - pi = %f - minDCF = %f" %(piT,Options['C'], pi,min_dcf_kfold))
                
        Options['rebalance']=False
        Options['C']=1e-1
        for pi in [0.1, 0.5, 0.9]:
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
        for piT in [0.1, 0.5, 0.9]:
            for pi in [0.1, 0.5, 0.9]:
                Options['C']=10
                Options['piT']=piT
                Options['gamma']=0.1
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
            Options['iterations']= 2
            _, scores_full, labels = validate.kfold(D, L, k, 0.5, gmm.compute_score, Options) #pi=0.5 actually not used to compute scores
           
            Options['Type']='diag'
            Options['iterations']= 2
            _, scores_diag, labels = validate.kfold(D, L, k, 0.5, gmm.compute_score, Options) #pi=0.5 actually not used to compute scores
            
            Options['Type']='full-tied'
            Options['iterations']= 2
            _, scores_full_tied, labels = validate.kfold(D, L, k, 0.5, gmm.compute_score, Options) #pi=0.5 actually not used to compute scores
            
            Options['Type']='diag-tied'
            Options['iterations']= 4
            _, scores_diag_tied, labels = validate.kfold(D, L, k, 0.5, gmm.compute_score, Options) #pi=0.5 actually not used to compute scores

        for pi in [0.1, 0.5, 0.9]:

            Options['Type']='full'
            Options['iterations']= 2
            min_dcf_kfold = validate.compute_min_DCF(scores_full, labels, pi, 1, 1)
            print(" gmm %s -components=%d - pi = %f --> minDCF = %f" %(Options['Type'], 2**Options['iterations'], pi,min_dcf_kfold))
            
            Options['Type']='diag'
            Options['iterations']= 2
            min_dcf_kfold = validate.compute_min_DCF(scores_diag, labels, pi, 1, 1)
            print(" gmm %s -components=%d - pi = %f --> minDCF = %f" %(Options['Type'], 2**Options['iterations'], pi,min_dcf_kfold))
            
            Options['Type']='full-tied'
            Options['iterations']= 2
            min_dcf_kfold = validate.compute_min_DCF(scores_full_tied, labels, pi, 1, 1)
            print(" gmm %s -components=%d - pi = %f --> minDCF = %f" %(Options['Type'], 2**Options['iterations'], pi,min_dcf_kfold))
            
            Options['Type']='diag-tied'
            Options['iterations']= 4
            min_dcf_kfold = validate.compute_min_DCF(scores_diag_tied, labels, pi, 1, 1)
            print(" gmm %s -components=%d - pi = %f --> minDCF = %f" %(Options['Type'], 2**Options['iterations'], pi,min_dcf_kfold))
                
        m = m-1
        D=D_copy #restore original dataset
            
   
          
def perform_calibration(D,L):
    calibration.min_vs_act(D, L)
    #calibration.optimal_threshold(D,L)
    #calibration.validate_score_trasformation(D, L)
    #calibration.min_vs_act_after_calibration(D, L)
  

def validate_fusion(D,L):
    fusion.validate_fused_scores(D,L)
    #fusion.ROC_with_fusion(D,L)
    #fusion.bayes_plot_with_fusion(D,L)


    
main()