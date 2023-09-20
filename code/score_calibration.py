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

    '''
    #GMM
    Options={
        'Type':'full-tied',
        'iterations': 2}   
    '''
    #Linear Logreg
    Options={
    'lambdaa' : 1e-5,
    'piT': 0.9,
    }
    if evaluation:
        #scores1 = gmm.compute_score(DEV, DTR, LTR, Options)
        scores1 = log_reg.compute_score(DEV, DTR, LTR, Options)

        y_min1, y_act1 = validate.bayes_error(pi_array, scores1, LEV)
    else:
        _ , scores1, labels1 = validate.kfold(DTR, LTR, 5, 1, log_reg.compute_score, Options) 
        y_min1, y_act1 = validate.bayes_error(pi_array, scores1, labels1)
        
    for pi in [0.1, 0.5, 0.9]:
        if evaluation:
            act_dcf = validate.compute_act_DCF(scores1, LEV, pi, 1, 1)
        else:
            act_dcf= validate.compute_act_DCF(scores1, labels1, pi, 1, 1)
        print("  Gaussian mixture model %s -components=%d - pi = %f --> actDCF = %f" %(Options['lambdaa'], 2**Options['piT'], pi,act_dcf))
    
    plt.figure()
    plt.plot(pi_array, y_min1, 'r--',  label='Linear LR min_DCF')
    plt.plot(pi_array, y_act1, 'r', label='Linear LR act_DCF')
    plt.legend()
    plt.ylim(top=1.5)
    plt.ylim(bottom=0)
    plt.xlabel("application")
    plt.ylabel("cost")
    plt.tight_layout() 
    if evaluation:
        plt.savefig("./images/ScoreCalibration/actVSmin_EVALUATION_logreg.png")
    else:
        plt.savefig("./images/ScoreCalibration/actVSmin.png")
    plt.show()
    

def compute_score_trasformation(scores_TR, LTR, scores_TE,pi):
    scores_TR= mrow(scores_TR)
    scores_TE= mrow(scores_TE)
    alfa, beta_prime = log_reg.train_log_reg(scores_TR, LTR, 1e-06, pi) 
    new_scores= numpy.dot(alfa.T,scores_TE)+beta_prime - numpy.log(pi/(1-pi))
    return new_scores
    
# trasform score so that theoretical threshold provide close to optimal values over different applications
def validate_score_trasformation(DTR,LTR, DEV=None, LEV=None, evaluation=False):
    print ('-------------------SCORE TRASFORMATION-------------------')
    
    '''
    #GMM
    Options={
        'Type':'full-tied',
        'iterations': 2}   
    '''
    #Linear Logreg
    Options={
    'lambdaa' : 1e-5,
    'piT': 0.9,
    }
     
     
    for pi in [0.1, 0.5, 0.9]:
        # _ , scores1, labels1 = validate.kfold(DTR, LTR, 5, 1, gmm.compute_score, Options)
        _ , scores1, labels1 = validate.kfold(DTR, LTR, 5, 1, log_reg.compute_score, Options)
        scores_TR1, LTR1, scores_TE1, LTE1 = split_scores(scores1, labels1) #split and shuffle scores
        calibrated_scores1 = compute_score_trasformation(scores_TR1, LTR1, scores_TE1, pi)
        if evaluation:
            #scores_ev = gmm.compute_score(DEV, DTR, LTR, Options) #GMM
            scores_ev = log_reg.compute_score(DEV, DTR, LTR, Options) #LOGREG

            calibrated_scores_ev = compute_score_trasformation(scores_TR1, LTR1, scores_ev, pi)

        min_DCF_validation = validate.compute_min_DCF(scores_TE1, LTE1, pi, 1, 1)
        act_DCF_non_calibrated= validate.compute_act_DCF(scores_TE1, LTE1, pi, 1, 1, th=None) 
        print('GMM full tied 4 components: prior pi= % f - min_dcf computed on validation scores = %f' %(pi,min_DCF_validation))
        print('GMM full tied 4 components: act DCF computed on theoretical threshold (pi=%f) without calibration = %f'%(pi,act_DCF_non_calibrated))
        if evaluation:
            min_DCF_ev = validate.compute_min_DCF(scores_ev, LEV, pi, 1, 1)
            act_DCF_non_calibrated_ev= validate.compute_act_DCF(scores_ev, LEV, pi, 1, 1, th=None) 
            print('Linear LR: prior pi= % f - min_dcf computed on evaluation scores = %f' %(pi,min_DCF_ev))
            print('Linear LR: act DCF computed on theoretical threshold (pi=%f) without calibration = %f for evaluation set'%(pi,act_DCF_non_calibrated_ev))
    
        print('GMM full tied 4 components: result of act_DCF computed on trasformed scores with log reg pi = %f reported below:'%pi)
        for pi in [0.1, 0.5, 0.9]:
            act_DCF= validate.compute_act_DCF(calibrated_scores1, LTE1, pi, 1, 1, th=None) 
            print('Linear LR: act DCF computed on theoretical threshold (pi=%f) but with trasformed scores = %f'%(pi,act_DCF))
            if evaluation:
                act_DCF_ev= validate.compute_act_DCF(calibrated_scores_ev, LEV, pi, 1, 1, th=None) 
                print('Linear LR: evaluation set act DCF computed on theoretical threshold (pi=%f) but with trasformed scores = %f'%(pi,act_DCF_ev))
        print('')
    

def min_vs_act_after_calibration(DTR,LTR, DEV=None, LEV=None, evaluation=False):
    print('-------------------- Bayes Error Plot after calibration started--------------------')
    
    pi_array = numpy.linspace(-4, 4, 100)
    
    '''Options={
        'Type':'full-tied',
        'iterations': 2}       
    '''
    #Linear Logreg
    Options={
    'lambdaa' : 1e-5,
    'piT': 0.9,
    }
     
    #_ , scores1, labels1 = validate.kfold(DTR, LTR, 5, 0.5, gmm.compute_score , Options)
    _ , scores1, labels1 = validate.kfold(DTR, LTR, 5, 0.5, log_reg.compute_score , Options)
    
    scores_TR1, LTR1, scores_TE1, LTE1 = split_scores(scores1, labels1) #split and shuffle scores
    calibrated_scores1 = compute_score_trasformation(scores_TR1, LTR1, scores_TE1, 0.5)
    y_min1, y_act1= validate.bayes_error(pi_array, calibrated_scores1, LTE1)
    if evaluation:
        #scores1_ev = gmm.compute_score(DEV, DTR, LTR, Options)
        scores1_ev = log_reg.compute_score(DEV, DTR, LTR, Options)
        calibrated_scores1 = compute_score_trasformation(scores_TR1, LTR1, scores1_ev, 0.5)
        y_min1, y_act1= validate.bayes_error(pi_array, calibrated_scores1, LEV)
    
    plt.figure()
    plt.plot(pi_array, y_min1, 'r--',  label='Linear LR min_DCF')
    plt.plot(pi_array, y_act1, 'r', label='Linear LR act_DCF')
    plt.legend()
    plt.ylim(top=1.5)
    plt.ylim(bottom=0)
    plt.xlabel("application")
    plt.ylabel("cost")
    plt.tight_layout() 
    plt.savefig("./images/ScoreCalibration/actVSmin_after_calibration_evaluation.png")
    plt.show()
    print('-------------- Bayes Error Plot after calibration ened -------------------')

def split_scores(D,L, seed=0):
    nTrain = int(D.shape[0]*8.0/10.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[0])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[idxTrain] 
    DTE = D[idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    
    return DTR, LTR, DTE, LTE
