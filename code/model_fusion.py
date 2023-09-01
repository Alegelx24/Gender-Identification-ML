import validate
import support_vector_machine as svm
import logistic_regression as log_reg
import score_calibration as calibration
import numpy
import gaussian_mixture_model as gmm
import matplotlib.pyplot as plt

def mrow(x):
    return numpy.reshape(x, (1,x.shape[0]))

def mcol(x):
    return numpy.reshape(x, (x.shape[0],1))


def fused_scores(D,L, scores_ev= None, evaluation= False):
    #best model1 
    Options={
        'Type':'full-tied',
        'iterations': 2}   
   
    _ , scores1, labels = validate.kfold(D, L, 5, 0.5, gmm.compute_score , Options)
    scores1_TR, LTR, scores1_TE, LTE = calibration.split_scores(scores1, labels) #split and shuffle scores
    scores1_TR = mrow(scores1_TR)
    scores1_TE = mrow(scores1_TE)
 
    #best model 2
    Options={
    'lambdaa' : 1e-5,
    'piT': 0.9,
    }  
    _ , scores2, labels = validate.kfold(D, L, 5, 1, log_reg.compute_score, Options) #pi(set to random value 1) actually not used to compute scores
    scores2_TR, LTR, scores2_TE, LTE = calibration.split_scores(scores2, labels) #same split used for best model 1 above. Labels returned are the same of best model 1
    scores2_TR = mrow(scores2_TR)
    scores2_TE = mrow(scores2_TE)
    
    all_scores_TR = numpy.vstack((scores1_TR, scores2_TR))
    all_scores_TE = numpy.vstack((scores1_TE, scores2_TE))
    pi=0.5
    alfa, beta_prime = log_reg.train_log_reg(all_scores_TR, LTR, 1e-06, pi) #??this pi should be fixed or not????
    new_scores= numpy.dot(alfa.T,all_scores_TE)+beta_prime - numpy.log(pi/(1-pi))
    if evaluation:
        new_scores = numpy.dot(alfa.T,scores_ev)+beta_prime - numpy.log(pi/(1-pi))
    return new_scores, LTE

def validate_fused_scores(D,L):
    for pi in [0.1, 0.5, 0.9]:
        fused_scores_TE, LTE = fused_scores(D,L)

        min_DCF_validation = validate.compute_min_DCF(fused_scores_TE, LTE, pi, 1, 1)
        print('Fusion: prior pi= % f - min_dcf computed on validation scores = %f' %(pi,min_DCF_validation))
        act_DCF_non_calibrated= validate.compute_act_DCF(fused_scores_TE, LTE, pi, 1, 1, th=None) #if th=None, compute_act_DCF function will use theoretical threshold
        print('act DCF computed on theoretical threshold (pi=%f) without calibration = %f'%(pi,act_DCF_non_calibrated))

def ROC_with_fusion(D,L):
    
    Options={
        'Type':'full-tied',
        'iterations': 2}   
    
    _ , scores1, labels1 = validate.kfold(D, L, 5, 1, gmm.compute_score, Options) #pi(set to random value 1) actually not used to compute scores
    scores_TR, LTR, scores_TE, LTE = calibration.split_scores(scores1, labels1) #split and shuffle scores
    calibrated_scores = calibration.score_trasformation(scores_TR, LTR, scores_TE, 0.5)
    FPR1, TPR1 =validate.ROC (calibrated_scores, LTE)
   
    Options={
    'lambdaa' : 1e-5,
    'piT': 0.9,
    }
    _ , scores2, labels2 = validate.kfold(D, L, 5, 1, log_reg.compute_score, Options) #pi(set to random value 1) actually not used to compute scores
    scores_TR, LTR, scores_TE, _ = calibration.split_scores(scores2, labels2) #split and shuffle scores
    calibrated_scores = calibration.score_trasformation(scores_TR, LTR, scores_TE, 0.5)
    FPR2, TPR2 =validate.ROC (calibrated_scores, LTE)
    
    fused_scores_TE, _ = fused_scores(D,L)
    FPR3, TPR3 =validate.ROC (fused_scores_TE, LTE)

    plt.figure()
    plt.plot(FPR1,TPR1, 'r', label = 'Gmm full tied', )
    plt.plot(FPR2,TPR2, 'b', label = 'Linear Log Reg')
    plt.plot(FPR3,TPR3, 'g', label = 'Fusion')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Negative Rate')
    plt.legend()
    plt.savefig("./images/fusion_ROC_gmm+lr.pdf" )
    plt.show()
    
def bayes_plot_with_fusion(D,L):
    pi_array = numpy.linspace(-4, 4, 20)

    Options={
        'Type':'full-tied',
        'iterations': 2}   
        
    _ , scores1, labels1 = validate.kfold(D, L, 5, 1, gmm.compute_score, Options) #pi(set to random value 1) actually not used to compute scores
    scores_TR, LTR, scores_TE, LTE = calibration.split_scores(scores1, labels1) #split and shuffle scores
    calibrated_scores = calibration.score_trasformation(scores_TR, LTR, scores_TE, 0.5)
    y_min1, y_act1 = validate.bayes_error(pi_array, calibrated_scores, LTE)
    for pi in [0.1, 0.5, 0.9]:
        act_dcf= validate.compute_act_DCF(scores1, labels1, pi, 1, 1)
        print ('Quad Log Reg: pi = %f --> act_DCF = %f'%(pi,act_dcf))
    
    Options={
    'lambdaa' : 1e-5,
    'piT': 0.9,
    }
    _ , scores2, labels2 = validate.kfold(D, L, 5, 1, log_reg.compute_score, Options) #pi(set to random value 1) actually not used to compute scores
    scores_TR, LTR, scores_TE, _ = calibration.split_scores(scores2, labels2) #split and shuffle scores
    calibrated_scores = calibration.score_trasformation(scores_TR, LTR, scores_TE, 0.5)
    y_min2, y_act2= validate.bayes_error(pi_array, calibrated_scores, LTE)
    for pi in [0.1, 0.5, 0.9]:
        act_dcf= validate.compute_act_DCF(scores2, labels2, pi, 1, 1)
        print ('RBF SVM: pi = %f --> act_DCF = %f'%(pi,act_dcf))
    
    fused_scores_TE, _ = fused_scores(D,L)
    y_min3, y_ac3t= validate.bayes_error(pi_array, fused_scores_TE, LTE)
    
    
    plt.figure()
    plt.plot(pi_array, y_min1, 'r',  label='GMM min_DCF')
    plt.plot(pi_array, y_min2, 'b',  label='Log Reg min_DCF')
    plt.plot(pi_array, y_min3, 'g',  label='Fusion min_DCF')
    plt.plot(pi_array, y_ac3t, 'g--',  label='Fusion act_DCF')
    plt.legend()
    plt.ylim(top=1.5)
    plt.ylim(bottom=0)
    plt.xlabel("application")
    plt.ylabel("cost")
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    plt.show()
    plt.savefig("./images/actVSmin_fusion_evaluatio_gmm+lr.pdf")
    
    


    
    