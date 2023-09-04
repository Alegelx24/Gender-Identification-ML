import validate
import numpy
import matplotlib.pyplot as plt
import logistic_regression as log_reg
import support_vector_machine as svm
import gaussian_mixture_model as gmm

def mrow(x):
    return numpy.reshape(x, (1,x.shape[0]))

def mcol(x):
    return numpy.reshape(x, (x.shape[0],1))


def min_vs_act(DTR,LTR, DEV=None, LEV=None, evaluation=False):
    pi_array = numpy.linspace(-4, 4, 100)

    Options={
        'Type':'full-tied',
        'iterations': 2}   
  
    if evaluation:
        scores1 = gmm.compute_score(DEV, DTR, LTR, Options)
        y_min1, y_act1 = validate.bayes_error(pi_array, scores1, LEV)
    else:
        _ , scores1, labels1 = validate.kfold(DTR, LTR, 5, 1, gmm.compute_score, Options) #pi(set to random value 1) actually not used to compute scores
        y_min1, y_act1 = validate.bayes_error(pi_array, scores1, labels1)
        
    for pi in [0.1, 0.5, 0.9]:
        if evaluation:
            act_dcf = validate.compute_act_DCF(scores1, LEV, pi, 1, 1)
        else:
            act_dcf= validate.compute_act_DCF(scores1, labels1, pi, 1, 1)
        print(" gmm %s -components=%d - pi = %f --> actDCF = %f" %(Options['Type'], 2**Options['iterations'], pi,act_dcf))
    
    plt.figure()
    plt.plot(pi_array, y_min1, 'r--',  label='Gmm Full Tied min_DCF')
    plt.plot(pi_array, y_act1, 'r', label='Gmm Full Tied act_DCF')
    plt.legend()
    plt.ylim(top=1.5)
    plt.ylim(bottom=0)
    plt.xlabel("application")
    plt.ylabel("cost")
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    if evaluation:
        plt.savefig("./images/ScoreCalibration/actVSmin_EVALUATION.pdf")
    else:
        plt.savefig("./images/ScoreCalibration/actVSmin.pdf")
    plt.show()
    
#first approach to calibrate the score: find optimal threshold on training scores samples, evaluate actual dcf with optimal threshold on evaluation scores samples
def optimal_threshold (DTR,LTR, DEV=None, LEV=None, evaluation=False):
    print('##############OPTIMAL THRESHOLD ESTIMATION #########################')
    
    #best model 1
    Options={
        'Type':'full-tied',
        'iterations': 2}   
   

    _ , scores, labels = validate.kfold(DTR, LTR, 5, 1, gmm.compute_score, Options) #pi(set to random value 1) actually not used to compute scores
    scores_TR1, LTR1, scores_TE1, LTE1 = split_scores(scores, labels) #split and shuffle scores
    t_array = numpy.arange (-100, 100, 0.1) #???
    if evaluation:
        scores_ev = gmm.compute_score(DEV, DTR, LTR, Options)
    for pi in [0.1, 0.5, 0.9]:
        threshold_act_DCF = {}
        for t in t_array:
            act_dcf= validate.compute_act_DCF(scores_TR1, LTR1, pi, 1, 1, th=t)
            threshold_act_DCF[t]=act_dcf
        best_t = min(threshold_act_DCF, key=threshold_act_DCF.get)
        print ('best threshold for pi=%f is t*=%f' %(pi, best_t))
        best_cost_validation = validate.compute_act_DCF(scores_TE1, LTE1, pi, 1, 1, th=best_t)
        theoretical_cost_validation = validate.compute_act_DCF(scores_TE1, LTE1, pi, 1, 1, th=None) #if th=None, computed_act_DCF function will use theoretical threshold
        min_DCF_validation = validate.compute_min_DCF(scores_TE1, LTE1, pi, 1, 1)
        print('GMM full tied 4 components: prior pi= % f - act_dcf computed on validation scores set for best threshold = %f' %(pi,best_cost_validation))
        print('GMM full tied 4 components prior pi= % f - act_dcf computed on validation scores set for theoretical threshold = %f' %(pi,theoretical_cost_validation))
        print('GMM full tied 4 components prior pi= % f - min_DCF computed on validation scores set = %f' %(pi,min_DCF_validation))
        if evaluation:
            best_cost_ev = validate.compute_act_DCF(scores_ev, LEV, pi, 1, 1, th=best_t)
            theoretical_cost_ev = validate.compute_act_DCF(scores_ev, LEV, pi, 1, 1, th=None) #if th=None, computed_act_DCF function will use theoretical threshold
            min_DCF_ev = validate.compute_min_DCF(scores_ev, LEV, pi, 1, 1)
            print('GMM full tied 4 components prior pi= % f - act_dcf computed on evaluation scores set for best threshold = %f' %(pi,best_cost_ev))
            print('GMM full tied 4 components prior pi= % f - act_dcf computed on evaluation scores set for theoretical threshold = %f' %(pi,theoretical_cost_ev))
            print('GMM full tied 4 components prior pi= % f - min_DCF computed on evaluation scores set = %f' %(pi,min_DCF_ev))
    '''
    #best model 2
    Options={
        'C' : 10,
        'piT': 0.5,
        'gamma':0.01,
        'rebalance':True
        }  
    _ , scores, labels = validate.kfold(DTR, LTR, 5, 1, svm.compute_score_RBF, Options) #pi(set to random value 1) actually not used to compute scores
    scores_TR2, LTR2, scores_TE2, LTE2 = split_scores(scores, labels) #split and shuffle scores
    if evaluation:
        scores_ev = svm.compute_score_RBF(DEV, DTR, LTR, Options)
    t_array = numpy.arange (-100, 100, 0.1) 
    for pi in [0.1, 0.5, 0.9]:
        threshold_act_DCF = {}
        for t in t_array:
            act_dcf= validate.compute_act_DCF(scores_TR2, LTR2, pi, 1, 1, th=t)
            threshold_act_DCF[t]=act_dcf
        best_t = min(threshold_act_DCF, key=threshold_act_DCF.get)
        best_cost_validation = validate.compute_act_DCF(scores_TE2, LTE2, pi, 1, 1, th=best_t)
        theoretical_cost_validation = validate.compute_act_DCF(scores_TE2, LTE2, pi, 1, 1, th=None) #if th=None, compute_act_DCF function will use theoretical threshold
        min_DCF_validation = validate.compute_min_DCF(scores_TE2, LTE2, pi, 1, 1)
        print('RBF SVM: prior pi= % f - act_dcf computed on validation scores set for best threshold = %f' %(pi,best_cost_validation))
        print('RBF SVM: prior pi= % f - act_dcf computed on validation scores set for theoretical threshold = %f' %(pi,theoretical_cost_validation))
        print('RBF SVM: prior pi= % f - min_dcf computed on validation scores = %f' %(pi,min_DCF_validation))
        if evaluation:
            best_cost_ev = validate.compute_act_DCF(scores_ev, LEV, pi, 1, 1, th=best_t)
            theoretical_cost_ev = validate.compute_act_DCF(scores_ev, LEV, pi, 1, 1, th=None) #if th=None, compute_act_DCF function will use theoretical threshold
            min_DCF_ev = validate.compute_min_DCF(scores_ev, LEV, pi, 1, 1)
            print('RBF SVM: prior pi= % f - act_dcf computed on evaluation scores set for best threshold = %f' %(pi,best_cost_ev))
            print('RBF SVM: prior pi= % f - act_dcf computed on evaluation scores set for theoretical threshold = %f' %(pi,theoretical_cost_ev))
            print('RBF SVM: prior pi= % f - min_dcf computed on evaluation scores = %f' %(pi,min_DCF_ev))
    '''
       
def score_trasformation(scores_TR, LTR, scores_TE,pi):
    scores_TR= mrow(scores_TR)
    scores_TE= mrow(scores_TE)
    alfa, beta_prime = log_reg.train_log_reg(scores_TR, LTR, 1e-06, pi) 
    new_scores= numpy.dot(alfa.T,scores_TE)+beta_prime - numpy.log(pi/(1-pi))
    return new_scores
    
#second approach to calibrate score: trasform score so that theoretical threshold provide close to optimal values over different applications
def validate_score_trasformation(DTR,LTR, DEV=None, LEV=None, evaluation=False):
    print ('\n\n############SCORE TRASFORMATION###################')
    #best model 1
    Options={
        'Type':'full-tied',
        'iterations': 2}   
     
    for pi in [0.1, 0.5, 0.9]:
        _ , scores1, labels1 = validate.kfold(DTR, LTR, 5, 1, gmm.compute_score, Options) #pi(set to random value 1) actually not used to compute scores
        scores_TR1, LTR1, scores_TE1, LTE1 = split_scores(scores1, labels1) #split and shuffle scores
        calibrated_scores1 = score_trasformation(scores_TR1, LTR1, scores_TE1, pi)
        if evaluation:
            scores_ev = gmm.compute_score(DEV, DTR, LTR, Options)
            calibrated_scores_ev = score_trasformation(scores_TR1, LTR1, scores_ev, pi)

        min_DCF_validation = validate.compute_min_DCF(scores_TE1, LTE1, pi, 1, 1)
        act_DCF_non_calibrated= validate.compute_act_DCF(scores_TE1, LTE1, pi, 1, 1, th=None) #if th=None, compute_act_DCF function will use theoretical threshold
        print('GMM full tied 4 components: prior pi= % f - min_dcf computed on validation scores = %f' %(pi,min_DCF_validation))
        print('GMM full tied 4 components: act DCF computed on theoretical threshold (pi=%f) without calibration = %f'%(pi,act_DCF_non_calibrated))
        if evaluation:
            min_DCF_ev = validate.compute_min_DCF(scores_ev, LEV, pi, 1, 1)
            act_DCF_non_calibrated_ev= validate.compute_act_DCF(scores_ev, LEV, pi, 1, 1, th=None) #if th=None, compute_act_DCF function will use theoretical threshold
            print('GMM full tied 4 components: prior pi= % f - min_dcf computed on evaluation scores = %f' %(pi,min_DCF_ev))
            print('GMM full tied 4 components: act DCF computed on theoretical threshold (pi=%f) without calibration = %f for evaluation set'%(pi,act_DCF_non_calibrated_ev))
    
        print('GMM full tied 4 components: result of act_DCF computed on trasformed scores with log reg pi = %f reported below:'%pi)
        for pi in [0.1, 0.5, 0.9]:
            act_DCF= validate.compute_act_DCF(calibrated_scores1, LTE1, pi, 1, 1, th=None) #if th=None, compute_act_DCF function will use theoretical threshold
            print('GMM full tied 4 components: act DCF computed on theoretical threshold (pi=%f) but with trasformed scores = %f'%(pi,act_DCF))
            if evaluation:
                act_DCF_ev= validate.compute_act_DCF(calibrated_scores_ev, LEV, pi, 1, 1, th=None) #if th=None, compute_act_DCF function will use theoretical threshold
                print('GMM full tied 4 components: evaluation set act DCF computed on theoretical threshold (pi=%f) but with trasformed scores = %f'%(pi,act_DCF_ev))
                
        print('')
    

def min_vs_act_after_calibration(DTR,LTR, DEV=None, LEV=None, evaluation=False):
    print('\n\n########## Bayes Error Plot after calibration START#################')
    
    pi_array = numpy.linspace(-4, 4, 100)
    
    Options={
        'Type':'full-tied',
        'iterations': 2}   
     
    _ , scores1, labels1 = validate.kfold(DTR, LTR, 5, 0.5, gmm.compute_score , Options)
    scores_TR1, LTR1, scores_TE1, LTE1 = split_scores(scores1, labels1) #split and shuffle scores
    calibrated_scores1 = score_trasformation(scores_TR1, LTR1, scores_TE1, 0.5)
    y_min1, y_act1= validate.bayes_error(pi_array, calibrated_scores1, LTE1)
    if evaluation:
        scores1_ev = gmm.compute_score(DEV, DTR, LTR, Options)
        calibrated_scores1 = score_trasformation(scores_TR1, LTR1, scores1_ev, 0.5)
        y_min1, y_act1= validate.bayes_error(pi_array, calibrated_scores1, LEV)
    
    plt.figure()
    plt.plot(pi_array, y_min1, 'r--',  label='GMM full tied min_DCF')
    plt.plot(pi_array, y_act1, 'r', label='GMM full tied act_DCF')
    plt.legend()
    plt.ylim(top=1.5)
    plt.ylim(bottom=0)
    plt.xlabel("application")
    plt.ylabel("cost")
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    plt.savefig("./images/ScoreCalibration/actVSmin_after_calibration_evaluation.pdf")
    plt.show()
    print('########## Bayes Error Plot after calibration END #################')

    

def split_scores(D,L, seed=0):
    nTrain = int(D.shape[0]*8.0/10.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[0]) #idx contains N numbers from 0 to N (where is equals to number of training samples) in a random  order
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[idxTrain] 
    DTE = D[idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    
    return DTR, LTR, DTE, LTE
