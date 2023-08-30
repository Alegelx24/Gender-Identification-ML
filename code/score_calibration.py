import validate
import numpy
import matplotlib.pyplot as plt
import logistic_regression as log_reg
import support_vector_machine as svm

def mrow(x):
    return numpy.reshape(x, (1,x.shape[0]))

def mcol(x):
    return numpy.reshape(x, (x.shape[0],1))


def min_vs_act(DTR,LTR, DEV=None, LEV=None, evaluation=False):
    pi_array = numpy.linspace(-4, 4, 20)

    Options={
    'lambdaa' : 1e-06,
    'piT': 0.1,
    }     
    if evaluation:
        scores1 = log_reg.compute_score_quadratic(DEV, DTR, LTR, Options)
        y_min1, y_act1 = validate.bayes_error(pi_array, scores1, LEV)
    else:
        _ , scores1, labels1 = validate.kfold(DTR, LTR, 5, 1, log_reg.compute_score_quadratic, Options) #pi(set to random value 1) actually not used to compute scores
        y_min1, y_act1 = validate.bayes_error(pi_array, scores1, labels1)
        
    for pi in [0.1, 0.5, 0.9]:
        if evaluation:
            act_dcf = validate.compute_act_DCF(scores1, LEV, pi, 1, 1)
        else:
            act_dcf= validate.compute_act_DCF(scores1, labels1, pi, 1, 1)
        print ('Quad Log Reg: pi = %f --> act_DCF = %f'%(pi,act_dcf))
    
    Options={
        'C' : 10,
        'piT': 0.5,
        'gamma':0.01,
        'rebalance':True
        }  
    if evaluation:
        scores2 = svm.compute_score_quadratic(DEV, DTR, LTR, Options)
        y_min2, y_act2= validate.bayes_error(pi_array, scores2, LEV)
    else:
        _ , scores2, labels2 = validate.kfold(DTR, LTR, 5, 1, svm.compute_score_RBF, Options) #pi(set to random value 1) actually not used to compute scores
        y_min2, y_act2= validate.bayes_error(pi_array, scores2, labels2)
    for pi in [0.1, 0.5, 0.9]:
        if evaluation:
            act_dcf = validate.compute_act_DCF(scores2, LEV, pi, 1, 1)
        else:
            act_dcf= validate.compute_act_DCF(scores2, labels2, pi, 1, 1)
        print ('RBF SVM: pi = %f --> act_DCF = %f'%(pi,act_dcf))
    
    plt.figure()
    plt.plot(pi_array, y_min1, 'r--',  label='Quad LogReg min_DCF')
    plt.plot(pi_array, y_act1, 'r', label='Quad Log Reg act_DCF')
    plt.plot(pi_array, y_min2, 'b--',  label='RBF SVM min_DCF')
    plt.plot(pi_array, y_act2, 'b', label='RBF SVM act_DCF')
    plt.legend()
    plt.ylim(top=1.5)
    plt.ylim(bottom=0)
    plt.xlabel("application")
    plt.ylabel("cost")
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    plt.show()
    if evaluation:
        plt.savefig("./images/ScoreCalibration/actVSmin_EVALUATION.pdf")
    else:
        plt.savefig("./images/ScoreCalibration/actVSmin.pdf")
    
#first approach to calibrate the score: find optimal threshold on training scores samples, evaluate actual dcf with optimal threshold on evaluation scores samples
def optimal_threshold (DTR,LTR, DEV=None, LEV=None, evaluation=False):
    print('##############OPTIMAL THRESHOLD ESTIMATION #########################')
    
    #best model 1
    Options={
    'lambdaa' : 1e-06,
    'piT': 0.1,
    }     

    _ , scores, labels = validate.kfold(DTR, LTR, 5, 1, log_reg.compute_score_quadratic, Options) #pi(set to random value 1) actually not used to compute scores
    scores_TR1, LTR1, scores_TE1, LTE1 = split_scores(scores, labels) #split and shuffle scores
    t_array = numpy.arange (-100, 100, 0.1) #To be changed ??
    if evaluation:
        scores_ev = log_reg.compute_score_quadratic(DEV, DTR, LTR, Options)
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
        print('Quad Log Reg: prior pi= % f - act_dcf computed on validation scores set for best threshold = %f' %(pi,best_cost_validation))
        print('Quad Log Reg: prior pi= % f - act_dcf computed on validation scores set for theoretical threshold = %f' %(pi,theoretical_cost_validation))
        print('Quad Log Reg: prior pi= % f - min_DCF computed on validation scores set = %f' %(pi,min_DCF_validation))
        if evaluation:
            best_cost_ev = validate.compute_act_DCF(scores_ev, LEV, pi, 1, 1, th=best_t)
            theoretical_cost_ev = validate.compute_act_DCF(scores_ev, LEV, pi, 1, 1, th=None) #if th=None, computed_act_DCF function will use theoretical threshold
            min_DCF_ev = validate.compute_min_DCF(scores_ev, LEV, pi, 1, 1)
            print('Quad Log Reg: prior pi= % f - act_dcf computed on evaluation scores set for best threshold = %f' %(pi,best_cost_ev))
            print('Quad Log Reg: prior pi= % f - act_dcf computed on evaluation scores set for theoretical threshold = %f' %(pi,theoretical_cost_ev))
            print('Quad Log Reg: prior pi= % f - min_DCF computed on evaluation scores set = %f' %(pi,min_DCF_ev))
   
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
    'lambdaa' : 1e-06,
    'piT': 0.1,
    }     
    for pi in [0.1, 0.5, 0.9]:
        _ , scores1, labels1 = validate.kfold(DTR, LTR, 5, 1, log_reg.compute_score_quadratic, Options) #pi(set to random value 1) actually not used to compute scores
        scores_TR1, LTR1, scores_TE1, LTE1 = split_scores(scores1, labels1) #split and shuffle scores
        calibrated_scores1 = score_trasformation(scores_TR1, LTR1, scores_TE1, pi)
        if evaluation:
            scores_ev = log_reg.compute_score_quadratic(DEV, DTR, LTR, Options)
            calibrated_scores_ev = score_trasformation(scores_TR1, LTR1, scores_ev, pi)

        min_DCF_validation = validate.compute_min_DCF(scores_TE1, LTE1, pi, 1, 1)
        act_DCF_non_calibrated= validate.compute_act_DCF(scores_TE1, LTE1, pi, 1, 1, th=None) #if th=None, compute_act_DCF function will use theoretical threshold
        print('Quad Log-Reg: prior pi= % f - min_dcf computed on validation scores = %f' %(pi,min_DCF_validation))
        print('Quad Log-Reg: act DCF computed on theoretical threshold (pi=%f) without calibration = %f'%(pi,act_DCF_non_calibrated))
        if evaluation:
            min_DCF_ev = validate.compute_min_DCF(scores_ev, LEV, pi, 1, 1)
            act_DCF_non_calibrated_ev= validate.compute_act_DCF(scores_ev, LEV, pi, 1, 1, th=None) #if th=None, compute_act_DCF function will use theoretical threshold
            print('Quad Log-Reg: prior pi= % f - min_dcf computed on evaluation scores = %f' %(pi,min_DCF_ev))
            print('Quad Log-Reg: act DCF computed on theoretical threshold (pi=%f) without calibration = %f for evaluation set'%(pi,act_DCF_non_calibrated_ev))
    
        print('Quad Log-Reg: result of act_DCF computed on trasformed scores with log reg pi = %f reported below:'%pi)
        for pi in [0.1, 0.5, 0.9]:
            act_DCF= validate.compute_act_DCF(calibrated_scores1, LTE1, pi, 1, 1, th=None) #if th=None, compute_act_DCF function will use theoretical threshold
            print('Quad Log-Reg: act DCF computed on theoretical threshold (pi=%f) but with trasformed scores = %f'%(pi,act_DCF))
            if evaluation:
                act_DCF_ev= validate.compute_act_DCF(calibrated_scores_ev, LEV, pi, 1, 1, th=None) #if th=None, compute_act_DCF function will use theoretical threshold
                print('Quad Log-Reg: evaluation set act DCF computed on theoretical threshold (pi=%f) but with trasformed scores = %f'%(pi,act_DCF_ev))
                
        print('')
    
    #best model 2
    Options={
        'C' : 10,
        'piT': 0.5,
        'gamma':0.01,
        'rebalance':True
        }  
    for pi in [0.1, 0.5, 0.9]:
        _ , scores2, labels2 = validate.kfold(DTR, LTR, 5, 1, svm.compute_score_RBF, Options) #pi(set to random value 1) actually not used to compute scores
        scores_TR2, LTR2, scores_TE2, LTE2 = split_scores(scores2, labels2) #split and shuffle scores
        calibrated_scores2 = score_trasformation(scores_TR2, LTR2, scores_TE2, pi)
        if evaluation:
            scores_ev = svm.compute_score_RBF(DEV, DTR, LTR, Options)
            calibrated_scores_ev = score_trasformation(scores_TR2, LTR2, scores_ev, pi)

        min_DCF_validation = validate.compute_min_DCF(scores_TE2, LTE2, pi, 1, 1)
        act_DCF_non_calibrated= validate.compute_act_DCF(scores_TE2, LTE2, pi, 1, 1, th=None) #if th=None, compute_act_DCF function will use theoretical threshold
        print('RBF SVM: prior pi= % f - min_dcf computed on validation scores = %f' %(pi,min_DCF_validation))
        print('RBF SVM: act DCF computed on theoretical threshold (pi=%f) without calibration = %f'%(pi,act_DCF_non_calibrated))
        if evaluation:
            min_DCF_ev = validate.compute_min_DCF(scores_ev, LEV, pi, 1, 1)
            act_DCF_non_calibrated_ev= validate.compute_act_DCF(scores_ev, LEV, pi, 1, 1, th=None) #if th=None, compute_act_DCF function will use theoretical threshold
            print('RBF SVM: prior pi= % f - min_dcf computed on evaluation scores = %f' %(pi,min_DCF_ev))
            print('RBF SVM: act DCF computed on theoretical threshold (pi=%f) without calibration evaluation = %f'%(pi,act_DCF_non_calibrated_ev))
            
        print('RBF SVM: result of act_DCF computed on trasformed scores with svm pi = %f reported below:'%pi)
        for pi in [0.1, 0.5, 0.9]:
            act_DCF= validate.compute_act_DCF(calibrated_scores2, LTE2, pi, 1, 1, th=None) #if th=None, compute_act_DCF function will use theoretical threshold
            print('RBF SVM: act DCF computed on theoretical threshold (pi=%f) but with trasformed scores = %f'%(pi,act_DCF))
            if evaluation:
                act_DCF_ev= validate.compute_act_DCF(calibrated_scores_ev, LEV, pi, 1, 1, th=None) #if th=None, compute_act_DCF function will use theoretical threshold
                print('RBF SVM: evaluation set act DCF computed on theoretical threshold (pi=%f) but with trasformed scores = %f'%(pi,act_DCF_ev))
                
        print('')

def min_vs_act_after_calibration(DTR,LTR, DEV=None, LEV=None, evaluation=False):
    print('\n\n########## Bayes Erro Plot after calibration START#################')
    
    pi_array = numpy.linspace(-4, 4, 20)
    
    Options={
    'lambdaa' : 1e-06,
    'piT': 0.1,
    }     
    _ , scores1, labels1 = validate.kfold(DTR, LTR, 5, 0.5, log_reg.compute_score_quadratic , Options)
    scores_TR1, LTR1, scores_TE1, LTE1 = split_scores(scores1, labels1) #split and shuffle scores
    calibrated_scores1 = score_trasformation(scores_TR1, LTR1, scores_TE1, 0.5)
    y_min1, y_act1= validate.bayes_error(pi_array, calibrated_scores1, LTE1)
    if evaluation:
        scores1_ev = log_reg.compute_score_quadratic(DEV, DTR, LTR, Options)
        calibrated_scores1 = score_trasformation(scores_TR1, LTR1, scores1_ev, 0.5)
        y_min1, y_act1= validate.bayes_error(pi_array, calibrated_scores1, LEV)
      
    Options={
        'C' : 10,
        'piT': 0.5,
        'gamma':0.01,
        'rebalance':True
        }  
    _ , scores2, labels2 = validate.kfold(DTR, LTR, 5, 1, svm.compute_score_RBF, Options) #pi(set to random value 1) actually not used to compute scores
    scores_TR2, LTR2, scores_TE2, LTE2 = split_scores(scores2, labels2) #split and shuffle scores
    calibrated_scores2 = score_trasformation(scores_TR2, LTR2, scores_TE2, 0.5)
    y_min2, y_act2= validate.bayes_error(pi_array, calibrated_scores2, LTE2)
    if evaluation:
        scores2_ev = svm.compute_score_RBF(DEV, DTR, LTR, Options)
        calibrated_scores2 = score_trasformation(scores_TR2, LTR2, scores2_ev, 0.5)
        y_min2, y_act2= validate.bayes_error(pi_array, calibrated_scores2, LEV)
    
    plt.figure()
    plt.plot(pi_array, y_min1, 'r--',  label='Quad Log Reg min_DCF')
    plt.plot(pi_array, y_act1, 'r', label='Quad Log Reg act_DCF')
    plt.plot(pi_array, y_min2, 'b--',  label='RBF SVM min_DCF')
    plt.plot(pi_array, y_act2, 'b', label='RBF SVM act_DCF')
    plt.legend()
    plt.ylim(top=1.5)
    plt.ylim(bottom=0)
    plt.xlabel("application")
    plt.ylabel("cost")
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    plt.show()
    plt.savefig("./images/ScoreCalibration/actVSmin_after_calibration.pdf")
    print('########## Bayes Erro Plot after calibration END #################')

    

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
