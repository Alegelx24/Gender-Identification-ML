import dataset_util as util
import validate as validate
import gaussian_classifier as gaussian
import logistic_regression as logreg
import support_vector_machine as svm
import gaussian_mixture_model as gmm
import score_calibration as calibration
import model_fusion as fusion
import evaluation as eval

k=5 #parameter K-fold

def main():
    D,L = util.load_training_dataset('./data/Train.txt')
    D_centered=util.center_dataset(D)[0]
    D_norm= util.perform_ZNormalization(D)[0]
        
    normalized= False 

    plot(D, L, normalized) 


    #TRAINING
    print("VALIDATION WITHOUT ZSCORE NORMALIZATION")

    train_gaussian_models(D, L)
   
    logreg.plot_minDCF_wrt_lambda(D, L, normalized=False)
    logreg.quadratic_plot_minDCF_wrt_lambda(D, L, normalized=False)
    train_logreg(D, L)
    
    svm.plot_linear_minDCF_wrt_C(D, L, normalized)
    svm.plot_quadratic_minDCF_wrt_C(D, L, normalized)
    svm.plot_RBF_minDCF_wrt_C(D, L, normalized)
    train_svm(D,L)

    gmm.plot_minDCF_wrt_components(D, D_norm, L)
    train_gmm(D, L)
      
    print("VALIDATION WITH ZSCORE NORMALIZATION")
    normalized= True 

    train_gaussian_models_zscore(D_norm, L)
   
    logreg.plot_minDCF_wrt_lambda(D_norm, L, normalized)
    logreg.quadratic_plot_minDCF_wrt_lambda(D_norm, L, normalized)
    train_logreg(D_norm, L)
    
    svm.plot_linear_minDCF_wrt_C(D_norm, L, normalized)
    svm.plot_quadratic_minDCF_wrt_C(D_norm, L, normalized)
    svm.plot_RBF_minDCF_wrt_C(D_norm, L, normalized)
    train_svm(D_norm,L)
    
    print("Calibration WITHOUT ZSCORE NORMALIZATION")
    perform_calibration(D,L)
    
    
    print("Calibration WITH ZSCORE NORMALIZATION")
    perform_calibration(D_norm,L)
    
    print("Fusion WITHOUT ZSCORE NORMALIZATION")
    validate_fusion(D,L)

    print("Fusion WITH ZSCORE NORMALIZATION")
    validate_fusion(D_norm,L)
    
    eval.perform_evaluation()

def plot(DTR, LTR, normalized):
    util.plot_fraction_explained_variance_pca(DTR, LTR)
    util.plot_histograms(DTR, LTR,normalized)
    util.plot_scatters(DTR, LTR,normalized)
    util.heatmap_generator(DTR, LTR, "correlation_heatmap")
    util.plotLDA(DTR, LTR,2)
    
def train_gaussian_models(D,L):
    Options={ }  
    D_copy = D
    m = 12
    while m>=7:
        if m < 12:
            D = util.pca(m, D)[0]
            print ("------ Gaussian classifiers with PCA(m = %d) ------" %m)
        else:
            print ("------ Gaussian classifiers with NO PCA ------")
        
        for pi in [0.1, 0.5, 0.9]:
            min_dcf_full = validate.kfold(D, L, k, pi, gaussian.compute_score_full, Options )[0]
            print(" Full-Cov MVG - pi = %f -> minDCF = %f" %(pi,min_dcf_full))
            min_dcf_diag = validate.kfold(D, L, k, pi, gaussian.compute_score_diagonal, Options )[0]
            print(" Diag-cov MVG - pi = %f -> minDCF = %f" %(pi,min_dcf_diag))
            min_dcf_tied_full = validate.kfold(D, L, k, pi, gaussian.compute_score_tied_full, Options )[0]
            print(" Tied full-cov MVG - pi = %f  minDCF = %f" %(pi,min_dcf_tied_full))
            min_dcf_tied_diag = validate.kfold(D, L, k, pi, gaussian.compute_score_tied_diagonal, Options )[0]
            print(" Tied diag-cov MVG - pi = %f  minDCF = %f" %(pi,min_dcf_tied_diag))

        m=m-1
        D=D_copy 


def train_gaussian_models_zscore(D,L):
    Options={ }  
    D_copy = D
    m = 12
    while m>=7:
        if m < 12:
            D = util.pca(m, D)[0]
            print ("------ Gaussian classifiers with PCA(m = %d) and Z-score norm ------" %m)
        else:
            print ("------ Gaussian classifiers with NO PCA and Z-score norm ------")
                    
        for pi in [0.1, 0.5, 0.9]:
            min_dcf_full = validate.kfold(D, L, k, pi, gaussian.compute_score_full, Options)[0]
            print(" Full-Cov - pi = %f -> minDCF = %f" %(pi,min_dcf_full))
            min_dcf_diag = validate.kfold(D, L, k, pi, gaussian.compute_score_diagonal, Options)[0]
            print(" Diag-cov - pi = %f -> minDCF = %f" %(pi,min_dcf_diag))
            min_dcf_tied_full = validate.kfold(D, L, k, pi, gaussian.compute_score_tied_full, Options)[0]
            print(" Tied full-cov - pi = %f  minDCF = %f" %(pi,min_dcf_tied_full))
            min_dcf_tied_diag = validate.kfold(D, L, k, pi, gaussian.compute_score_tied_diagonal, Options)[0]
            print(" Tied diag-cov - pi = %f  minDCF = %f" %(pi,min_dcf_tied_diag))

        m=m-1
        D=D_copy 
    
        
def train_logreg(D,L):
    D_copy = D
    Options={
    'lambdaa' : None,
    'piT': None,
    }  
    m = 12
    while m>=9:
        
        if m < 12:
            D = util.pca(m, D)[0]
            print ("------ Linear Logistic regression with PCA(m = %d) -------" %m)
        else:
            print ("------ Linear Logistic regression with NO PCA ------")
            
        for piT in [0.1, 0.5, 0.9]:
            Options['lambdaa']=1e-05    
            Options['piT']=piT
            for pi in [0.1, 0.5, 0.9]:
                min_dcf_kfold = validate.kfold(D, L, k, pi, logreg.compute_score, Options)[0]
                print(" Logistic regression -piT = %f - pi = %f  minDCF = %f" %(Options['piT'], pi,min_dcf_kfold))
             
        if m < 12:
            D = util.pca(m, D)[0]
            print ("------ Quadratic Logistic regression with PCA(m = %d) ------" %m)
        else:
            print ("------ Quadratic Logistic regression with NO PCA ------")
            
        for piT in [0.1, 0.5, 0.9]:
            Options['lambdaa']=100
            Options['piT']=piT
            for pi in [0.1, 0.5, 0.9]:
                min_dcf_kfold = validate.kfold(D, L, k, pi, logreg.compute_score_quadratic, Options)[0]
                print(" Quadratic Log Reg -piT = %f - pi = %f  minDCF = %f" %(Options['piT'], pi,min_dcf_kfold))  
        
        m = m-1
        D=D_copy 
    


def train_svm(D,L):
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
            print ("------- SVM LINEAR with PCA(m = %d) -------" %m)
        else:
            print ("------ SVM LINEAR with NO PCA -------")
            
        Options['C']=1
        for piT in [0.1, 0.5, 0.9]:
            for pi in [0.1, 0.5, 0.9]:
                Options['piT']=piT
                Options['rebalance']=True
                min_dcf_kfold = validate.kfold(D, L, k, pi, svm.compute_score_SVM_linear, Options)[0]
                print("Linear SVM with -piT = %f -C=%f - pi = %f - minDCF = %f" %(piT,Options['C'], pi,min_dcf_kfold))
                
        Options['rebalance']=False
        for pi in [0.1, 0.5, 0.9]:
                min_dcf_kfold = validate.kfold(D, L, k, pi, svm.compute_score_SVM_linear, Options)[0]
                print("Linear SVM without rebalancing -C=%f - pi = %f - minDCF = %f" %(Options['C'], pi,min_dcf_kfold)) 
          

        if m < 12:
            D = util.pca(m, D)[0]
            print ("------ SVM QUADRATIC with PCA(m = %d) ------" %m)
        else:
            print ("------ SVM QUADRATIC with NO PCA ------")
        for piT in [0.5]:
            for pi in [0.5, 0.9]:
                Options['C']=0.1
                Options['piT']=piT
                Options['rebalance']=True
                min_dcf_kfold = validate.kfold(D, L, k, pi, svm.compute_score_SVM_quadratic, Options)[0]
                print("Quadratric SVM with -piT = %f -C=%f - pi = %f - minDCF = %f" %(piT,Options['C'], pi,min_dcf_kfold))
                
        Options['rebalance']=False
        Options['C']=1e-1
        for pi in [0.1, 0.5, 0.9]:
            min_dcf_kfold = validate.kfold(D, L, k, pi, svm.compute_score_SVM_quadratic, Options)[0]
            print("Quadratic SVM without rebalancing -C=%f - pi = %f - minDCF = %f" %(Options['C'], pi,min_dcf_kfold))
           

        if m < 12:
            D = util.pca(m, D)[0]
            print ("------ SVM RBF with PCA(m = %d) ------" %m)
        else:
            print ("------ SVM RBF with NO PCA ------")
        for piT in [0.1, 0.5, 0.9]:
            for pi in [0.1, 0.5, 0.9]:
                Options['C']=10
                Options['piT']=piT
                Options['gamma']=0.1
                Options['rebalance']=True
                min_dcf_kfold = validate.kfold(D, L, k, pi, svm.compute_score_SVM_RBF, Options)[0]
                print("RBF SVM with -piT = %f -gamma =%f -C=%f - pi = %f -> minDCF = %f" %(piT, Options['gamma'], Options['C'], pi,min_dcf_kfold))      
            
        Options['rebalance']=False
        for pi in [0.1, 0.5, 0.9]:
            pass
            min_dcf_kfold = validate.kfold(D, L, k, pi, svm.compute_score_SVM_RBF, Options)[0]
            print("RBF SVM without rebalancing -gamma =%f -C=%f - pi = %f -> minDCF = %f" %(Options['gamma'], Options['C'], pi,min_dcf_kfold))
              
        m = m-1
        D=D_copy 


        
def train_gmm(D,L):
    D_copy = D
    Options={ 
        'Type':None,
        'iterations':None #keep in mind that gmm components = 2^#iterations
        }  
    m = 12
    while m>=9:
        if m < 12:
            D = util.pca(m, D)[0]
            print ("------ GMM with PCA(m = %d) ------" %m)
        else:
            print ("------ GMM with NO PCA ------")
            
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
        D=D_copy 
            
   
          
def perform_calibration(D,L):
    calibration.min_vs_act(D, L)
    calibration.optimal_threshold(D,L)
    calibration.validate_score_trasformation(D, L)
    calibration.min_vs_act_after_calibration(D, L)
  

def validate_fusion(D,L):
    fusion.validate_fused_scores(D,L)
    fusion.plot_ROC_with_fusion(D,L)
    fusion.bayes_plot_with_fusion(D,L)


    
main()