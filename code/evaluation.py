import dataset_util as util
import gaussian_classifier as gauss
import gaussian_mixture_model as gmm
import validate
import logistic_regression as log_reg
import support_vector_machine as svm
import numpy
import score_calibration as calibration
import matplotlib.pyplot as plt
import model_fusion as fusion 

def evaluation():

    DTR,LTR = util.load_training_set('../Data/Train.txt')
    DEV, LEV = util.load_evaluation_set('../Data/Test.txt')
    DTR,DEV= util.scale_ZNormalization(DTR, DEV, normalize_ev = True) #Normalize evaluation samples using mean and covariance from training set
    DTR_gaussianized= util.gaussianize_training(DTR)
    DEV_gaussianized = util.gaussianize_evaluation(DEV, DTR) #Gaussianzize evaluation samples comparing them with training set
    
    print('-----------EVALUATION WITH RAW FEATURES STARTED...-----------------')
    gaussianize=False
    evaluation_MVG(DTR, LTR, DEV, LEV)
    evaluation_log_reg(DTR, LTR, DEV, LEV, gaussianize)
    evaluation_SVM(DTR, LTR, DEV, LEV, gaussianize)
    evaluation_gmm(DTR, DTR_gaussianized, LTR, DEV, DEV_gaussianized, LEV, gaussianize)

    
    print('-----------EVALUATION ON GAUSSIANIZED FEATURES STARTED...-----------------')
    gaussianize=True
    evaluation_MVG(DTR_gaussianized, LTR, DEV_gaussianized, LEV)
    evaluation_log_reg(DTR_gaussianized, LTR, DEV_gaussianized, LEV, gaussianize)
    evaluation_SVM(DTR_gaussianized, LTR, DEV_gaussianized, LEV,gaussianize )
    evaluation_gmm(DTR, DTR_gaussianized, LTR, DEV, DEV_gaussianized, LEV, gaussianize)

    
    calibration.min_vs_act(DTR,LTR, DEV=DEV, LEV=LEV, evaluation=True)
    calibration.optimal_threshold(DTR, LTR, DEV=DEV, LEV=LEV, evaluation=True)
    calibration.validate_score_trasformation(DTR,LTR, DEV=DEV, LEV=LEV, evaluation=True)
    calibration.min_vs_act_after_calibration(DTR, LTR, DEV=DEV, LEV=LEV, evaluation=True)
    
    evaluation_fusion(DTR, LTR, DEV, LEV)


def evaluation_MVG(DTR, LTR, DEV, LEV):
    DTR_copy = DTR
    DEV_copy = DEV
    Options={ }  
    m = 8
    while m>=5:
        if m < 8:
            DTR, P = util.pca(m, DTR)
            DEV = numpy.dot(P.T, DEV)
            print ("##########################################")
            print ("##### Gaussian classifiers with m = %d ####" %m)
            print ("##########################################")
        else:
            print ("##########################################")
            print ("#### Gaussian classifiers with NO PCA ####")
            print ("##########################################")
            
        #Train models on all the training data and compute scores for the evaluation dataset
        scores_full = gauss.compute_score_full(DEV,DTR,LTR,Options) 
        scores_diag = gauss.compute_score_diag(DEV,DTR,LTR,Options) 
        scores_full_tied = gauss.compute_score_tied_full(DEV,DTR,LTR,Options) 
        scores_tied_diag = gauss.compute_score_tied_diag(DEV,DTR,LTR,Options) 
        for pi in [0.1, 0.5, 0.9]:
            #compute min DCF on evaluation set
            min_DCF = validate.compute_min_DCF(scores_full, LEV, pi, 1, 1)
            print("full covariance gaussian raw features no PCA - pi = %f --> min_DCF= %f" %(pi,min_DCF))
            
            min_DCF = validate.compute_min_DCF(scores_diag, LEV, pi, 1, 1)
            print("diag covariance gaussian raw features no PCA - pi = %f --> min_DCF= %f" %(pi,min_DCF))
            
            min_DCF = validate.compute_min_DCF(scores_full_tied, LEV, pi, 1, 1)
            print("full-tied covariance gaussian raw features no PCA - pi = %f --> min_DCF= %f" %(pi,min_DCF))
            
            min_DCF = validate.compute_min_DCF(scores_tied_diag, LEV, pi, 1, 1)
            print("tied-diag covariance gaussian raw features no PCA - pi = %f --> min_DCF= %f" %(pi,min_DCF))
        
        m=m-1
        DEV = DEV_copy
        DTR = DTR_copy
        

def evaluation_log_reg(DTR, LTR, DEV, LEV, gaussianize):
    log_reg.plot_minDCF_wrt_lamda(DTR,LTR, gaussianize, DEV=DEV, LEV=LEV, evaluation=True)
    log_reg.quadratic_plot_minDCF_wrt_lamda(DTR, LTR, gaussianize, DEV=DEV, LEV=LEV, evaluation=True)
    DTR_copy = DTR
    DEV_copy = DEV
    m = 8
    while m>=5:
        if m < 8:
            DTR, P = util.pca(m, DTR)
            DEV = numpy.dot(P.T, DEV)
            print ("##########################################")
            print ("##### Logistic Regression with m = %d ####" %m)
            print ("##########################################")
        else:
            print ("##########################################")
            print ("#### Logistic Regression with NO PCA ####")
            print ("##########################################")
            
        Options={
        'lambdaa' : 1e-6,
        'piT': None,
        }              
        for piT in [0.1, 0.5, 0.9]:
            Options['piT']=piT
            scores_linear_log_reg = log_reg.compute_score(DEV, DTR, LTR, Options)
            scores_quadratic_log_reg = log_reg.compute_score_quadratic(DEV, DTR, LTR, Options)
            for pi in [0.1, 0.5, 0.9]:
                min_DCF = validate.compute_min_DCF(scores_linear_log_reg, LEV, pi, 1, 1)
                print("linear log reg -lamda = 10^-6 -piT=%f - pi = %f --> min_DCF= %f" %(piT, pi,min_DCF))
                min_DCF = validate.compute_min_DCF(scores_quadratic_log_reg, LEV, pi, 1, 1)
                print("Quadratic log reg -lamda = 10^-6 -piT=%f - pi = %f --> min_DCF= %f" %(piT, pi,min_DCF))
        
        m=m-1
        DEV = DEV_copy
        DTR = DTR_copy
        
        
def evaluation_SVM(DTR, LTR, DEV, LEV, gaussianize):
    svm.plot_linear_minDCF_wrt_C(DTR,LTR,gaussianize, DEV=DEV, LEV=LEV, evaluation=True)
    svm.plot_quadratic_minDCF_wrt_C(DTR,LTR,gaussianize, DEV=DEV, LEV=LEV, evaluation=True)
    svm.plot_RBF_minDCF_wrt_C(DTR,LTR,gaussianize, DEV=DEV, LEV=LEV, evaluation=True)
    Options={
        'C' : None,
        'piT': None,
        'gamma':None,
        'rebalance':None
        }  
    DTR_copy = DTR
    DEV_copy = DEV
    m = 8
    while m>=5:
        if m < 8:
            DTR, P = util.pca(m, DTR)
            DEV = numpy.dot(P.T, DEV)
            print ("##########################################")
            print ("##### SVM with m = %d ####" %m)
            print ("##########################################")
        else:
            print ("##########################################")
            print ("#### SVM with NO PCA ####")
            print ("##########################################")
            
        for piT in [0.1, 0.5, 0.9]:
            Options['C']=1
            Options['piT']=piT
            Options['rebalance']=True
            scores_linear_svm = svm.compute_score_linear(DEV, DTR, LTR, Options)
            for pi in [0.1, 0.5, 0.9]:
                min_DCF = validate.compute_min_DCF(scores_linear_svm, LEV, pi, 1, 1)
                print("linear SVM -C =1 -piT=%f - pi = %f --> min_DCF= %f" %(piT, pi,min_DCF))
        
        Options['C']=1
        Options['rebalance']=False
        scores_linear_svm = svm.compute_score_linear(DEV, DTR, LTR, Options)
        for pi in [0.1, 0.5, 0.9]:
            min_DCF = validate.compute_min_DCF(scores_linear_svm, LEV, pi, 1, 1)
            print("linear SVM -C =1 -No rebalancing - pi = %f --> min_DCF= %f" %(pi,min_DCF))
            
        if m < 8:
            DTR, P = util.pca(m, DTR)
            DEV = numpy.dot(P.T, DEV)
            print ("##########################################")
            print ("##### SVM Quadratci with m = %d ####" %m)
            print ("##########################################")
        else:
            print ("##########################################")
            print ("#### SVM Quadratic with NO PCA ####")
            print ("##########################################")
            
        for piT in [0.1, 0.5, 0.9]:
            Options['C']=0.1
            Options['piT']=piT
            Options['rebalance']=True
            scores_quadratic_svm = svm.compute_score_quadratic(DEV, DTR, LTR, Options)
            for pi in [0.1, 0.5, 0.9]:
                min_DCF = validate.compute_min_DCF(scores_quadratic_svm, LEV, pi, 1, 1)
                print("linear SVM -C =1 -piT=%f - pi = %f --> min_DCF= %f" %(piT, pi,min_DCF))
        
        Options['C']=0.1
        Options['rebalance']=False
        scores_quadratic_svm = svm.compute_score_quadratic(DEV, DTR, LTR, Options)
        for pi in [0.1, 0.5, 0.9]:
            min_DCF = validate.compute_min_DCF(scores_quadratic_svm, LEV, pi, 1, 1)
            print("linear SVM -C =1 -No rebalancing - pi = %f --> min_DCF= %f" %(pi,min_DCF))
            
        if m < 8:
            DTR, P = util.pca(m, DTR)
            DEV = numpy.dot(P.T, DEV)
            print ("##########################################")
            print ("##### SVM RBF with m = %d ####" %m)
            print ("##########################################")
        else:
            print ("##########################################")
            print ("#### SVM RBF with NO PCA ####")
            print ("##########################################")
            
        for piT in [0.1, 0.5, 0.9]:
            Options['C']=10
            Options['piT']=piT
            Options['gamma']=0.01
            Options['rebalance']=True
            scores_rbf_svm = svm.compute_score_RBF(DEV, DTR, LTR, Options)
            for pi in [0.1, 0.5, 0.9]:
                min_DCF = validate.compute_min_DCF(scores_rbf_svm, LEV, pi, 1, 1)
                print("linear SVM -C =1 -piT=%f - pi = %f --> min_DCF= %f" %(piT, pi,min_DCF))
        
        Options['C']=10
        Options['rebalance']=False
        scores_rbf_svm = svm.compute_score_RBF (DEV, DTR, LTR, Options)
        for pi in [0.1, 0.5, 0.9]:
            min_DCF = validate.compute_min_DCF(scores_rbf_svm, LEV, pi, 1, 1)
            print("linear SVM -C =1 -No rebalancing - pi = %f --> min_DCF= %f" %(pi,min_DCF))    
        
        m=m-1
        DEV = DEV_copy
        DTR = DTR_copy


def evaluation_gmm(DTR, DTR_gaussianized, LTR, DEV, DEV_gaussianized, LEV, gaussianize):
    gmm.plot_minDCF_wrt_components(DTR, DTR_gaussianized, LTR, DEV=DEV, DEV_gaussianized=DEV_gaussianized, LEV=LEV, evaluation=True  )
    DTR_copy = DTR
    DEV_copy = DEV
    Options={ 
        'Type':None,
        'iterations':None #components will be 2^iterations
        }  
    m = 8
    if gaussianize:
        DTR = DTR_gaussianized
        DEV = DEV_gaussianized
    while m>=5:
        if m < 8:
            DTR, P = util.pca(m, DTR)
            DEV = numpy.dot(P.T, DEV)
            print ("##########################################")
            print ("############# GMM with m = %d #############" %m)
            print ("##########################################")
        else:
            print ("##########################################")
            print ("############ GMM with NO PCA #############")
            print ("##########################################")
            
        #Train models on all the training data and compute scores for the evaluation dataset
        Options['Type']='full'
        Options['iterations']= 4
        scores_full = gmm.compute_score(DEV,DTR,LTR,Options) 
        Options['Type']='diag'
        Options['iterations']= 5
        scores_diag = gmm.compute_score(DEV,DTR,LTR,Options)  
        Options['Type']='full-tied'
        Options['iterations']= 6
        scores_full_tied = gmm.compute_score(DEV,DTR,LTR,Options)  
        Options['Type']='diag-tied'
        Options['iterations']= 6
        scores_tied_diag = gmm.compute_score(DEV,DTR,LTR,Options) 
        for pi in [0.1, 0.5, 0.9]:
            #compute min DCF on evaluation set
            min_DCF = validate.compute_min_DCF(scores_full, LEV, pi, 1, 1)
            print(" gmm full -components=16 - pi = %f --> minDCF = %f" %( pi,min_DCF))

            min_DCF = validate.compute_min_DCF(scores_diag, LEV, pi, 1, 1)
            print(" gmm diag -components=32 - pi = %f --> minDCF = %f" %( pi,min_DCF))
            
            min_DCF = validate.compute_min_DCF(scores_full_tied, LEV, pi, 1, 1)
            print(" gmm full-tied -components=64 - pi = %f --> minDCF = %f" %( pi,min_DCF))
            
            min_DCF = validate.compute_min_DCF(scores_tied_diag, LEV, pi, 1, 1)
            print(" gmm diag-tied -components=64 - pi = %f --> minDCF = %f" %( pi,min_DCF))        
    
        m=m-1
        DEV = DEV_copy
        DTR = DTR_copy


def evaluation_fusion(DTR, LTR, DEV, LEV):
    Options={
    'lambdaa' : 1e-06,
    'piT': 0.1,
    }   
    scores1 = log_reg.compute_score_quadratic(DEV, DTR, LTR, Options)
    scores1 = util.mrow(scores1)
 
    Options={
        'C' : 10,
        'piT': 0.5,
        'gamma':0.01,
        'rebalance':True
        }  
    scores2 = svm.compute_score_RBF(DEV, DTR, LTR, Options)
    scores2 = util.mrow(scores2)
    
    all_scores_ev = numpy.vstack((scores1, scores2))
    
    fused_scores, _ = fusion.fused_scores(DTR, LTR, scores_ev=all_scores_ev, evaluation = True)
    for pi in [0.1, 0.5, 0.9]:
        min_DCF = validate.compute_min_DCF(fused_scores, LEV, pi, 1, 1)
        print("Fusion - pi = %f --> min_DCF= %f" %(pi,min_DCF))
        act_DCF = validate.compute_act_DCF(fused_scores, LEV, pi, 1, 1)
        print("Fusion - pi = %f --> act_DCF= %f" %(pi,act_DCF))

    bayes_plot_with_fusion_evaluation(DTR, LTR, DEV, LEV)
    
    
def bayes_plot_with_fusion_evaluation(DTR, LTR, DEV, LEV):
    pi_array = numpy.linspace(-4, 4, 20)

    #Model 1
    Options={
    'lambdaa' : 1e-06,
    'piT': 0.1,
    }     
    _ , scores1, labels1 = validate.kfold(DTR, LTR, 5, 0.5, log_reg.compute_score_quadratic , Options)
    scores_TR1, LTR1, scores_TE1, LTE1 = calibration.split_scores(scores1, labels1) #split and shuffle scores
    scores1_ev = log_reg.compute_score_quadratic(DEV, DTR, LTR, Options)
    calibrated_scores1 = calibration.score_trasformation(scores_TR1, LTR1, scores1_ev, 0.5)
    y_min1, y_act1= validate.bayes_error(pi_array, calibrated_scores1, LEV)
    
    #Model 2
    Options={
        'C' : 10,
        'piT': 0.5,
        'gamma':0.01,
        'rebalance':True
        }  
    _ , scores2, labels2 = validate.kfold(DTR, LTR, 5, 1, svm.compute_score_RBF, Options) #pi(set to random value 1) actually not used to compute scores
    scores_TR2, LTR2, scores_TE2, LTE2 = calibration.split_scores(scores2, labels2) #split and shuffle scores
    scores2_ev = svm.compute_score_RBF(DEV, DTR, LTR, Options)
    calibrated_scores2 = calibration.score_trasformation(scores_TR2, LTR2, scores2_ev, 0.5)
    y_min2, y_act2= validate.bayes_error(pi_array, calibrated_scores2, LEV)
    
    #Fusion
    scores1_ev = util.mrow(scores1_ev)
    scores2_ev = util.mrow(scores2_ev)
    all_scores_ev = numpy.vstack((scores1_ev, scores2_ev))
    fused_scores_ev, _ = fusion.fused_scores(DTR, LTR, scores_ev=all_scores_ev, evaluation = True)
    y_min3, y_ac3t= validate.bayes_error(pi_array, fused_scores_ev, LEV)
    
    #Plot Bayes error
    plt.figure()
    plt.plot(pi_array, y_min1, 'r',  label='Quad LogReg min_DCF')
    plt.plot(pi_array, y_min2, 'b',  label='RBF SVM min_DCF')
    plt.plot(pi_array, y_min3, 'g',  label='Fusion min_DCF')
    plt.plot(pi_array, y_ac3t, 'g--',  label='Fusion act_DCF')
    plt.legend()
    plt.ylim(top=1.5)
    plt.ylim(bottom=0)
    plt.xlabel("application")
    plt.ylabel("cost")
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    plt.show()
    plt.savefig("../Images/actVSmin_fusion_evaluation.pdf")
    
   
    
    
   