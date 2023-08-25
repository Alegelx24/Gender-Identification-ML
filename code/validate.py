import numpy
import matplotlib.pyplot as plt
import pylab


def assign_labels(scores, pi, Cfn, Cfp, th=None):
    if th is None:
        th= - numpy.log(pi*Cfn) + numpy.log((1-pi)*Cfp)
    P= scores > th
    return numpy.int32(P)
        

def compute_confusion_matrix_binary(Pred, Labels):
    C=numpy.zeros((2,2))
    C[0,0] = ((Pred==0) * (Labels==0)).sum()
    C[0,1] = ((Pred==0) * (Labels==1)).sum()
    C[1,0] = ((Pred==1) * (Labels==0)).sum()
    C[1,1] = ((Pred==1) * (Labels==1)).sum()
    return C


def compute_emp_Bayes_binary(CM, pi, Cfn, Cfp):
    fnr= CM[0,1] / (CM[0,1]+CM[1,1])
    fpr= CM[1,0] / (CM[0,0]+CM[1,0])
    return pi*Cfn*fnr + (1-pi) * Cfp * fpr


def compute_normalized_emp_Bayes(CM, pi, Cfn, Cfp):
    empBayes= compute_emp_Bayes_binary(CM, pi, Cfn, Cfp)
    return empBayes/ min(pi*Cfn, (1-pi)*Cfp)
    

def compute_act_DCF(scores, labels, pi, Cfn, Cfp, th=None):
    Pred= assign_labels(scores, pi, Cfn, Cfp, th=th)
    CM= compute_confusion_matrix_binary(Pred, labels)
    return compute_normalized_emp_Bayes(CM, pi, Cfn, Cfp)


def compute_min_DCF(scores, labels, pi, Cfn, Cfp):
    t = numpy.array(scores)
    t.sort()
    numpy.concatenate([numpy.array([-numpy.inf]), t, numpy.array([numpy.inf]) ])
    dcfList = []
    for _th in t:
        dcfList.append(compute_act_DCF(scores, labels, pi, Cfn, Cfp, th=_th))
    return numpy.array(dcfList).min()


def bayes_error(pArray, scores, labels):
    y_min=[]
    y_act=[]
    for p in pArray:
        pi=1.0/(1.0+numpy.exp(-p))
        y_min.append(compute_min_DCF(scores, labels, pi, 1, 1))
        y_act.append(compute_act_DCF(scores, labels, pi, 1, 1)) 
    return y_min, y_act
  

def ROC(scores, Labels):
    
    #The ROC curve require to compute and plot TPR and FPR changing threshold. I have to sweep all the possible trhesholds.
    #I consider all the thresholds corresponding to mhy sorted scores.
    thresholds = numpy.array(scores)
    thresholds.sort()
    thresholds = numpy.concatenate([numpy.array([-numpy.inf]), thresholds, numpy.array([numpy.inf]) ]) #I add -infinity to te beginning and +infinity to the end
    
    #I create an array for FPR and TPR, each element of these array correspond to FPR and TPR for a certain threshold
    FPR = numpy.zeros(thresholds.size)
    TPR = numpy.zeros(thresholds.size)
    for idx,t in enumerate(thresholds):
        #For each value of a trheshold t I compute a new Confusion matrix from which I extract TPR and FPR
        Pred= numpy.int32(scores>t)
        Conf = numpy.zeros((2,2))
        for i in range(2):
            for j in range(2):
                Conf[i,j] =((Pred==i) * (Labels==j)).sum()
            TPR[idx]= Conf[1,1] / (Conf[1,1]+Conf[0,1])
            FPR[idx]= Conf[1,0] / (Conf[1,0]+Conf[0,0])
            
    return FPR,TPR


def kfold(D,L, k, pi, compute_s, Options):
    numpy.random.seed(0)
    indexes = numpy.random.permutation(D.shape[1]) #Shuffle
    fold_dim  = int(D.shape[1]/k)
    scores = numpy.array([])
    labels = numpy.array([]) #if fold_dim is a float number labels < L
    
    for i in range(k):
        idx_test = indexes[i*fold_dim:(i+1)*fold_dim]
        
        idx_train_left = indexes [0 : i*fold_dim]
        idx_train_right = indexes [(i+1)*fold_dim :]
        idx_train = numpy.hstack([idx_train_left, idx_train_right])
        
        DTR = D[:,idx_train]
        LTR = L[idx_train]
        DTE = D[:,idx_test]
        LTE = L[idx_test]
        
        score_ith = compute_s(DTE,DTR,LTR, Options)
        scores = numpy.append(scores, score_ith)
        labels = numpy.append(labels, LTE)  
    min_dcf = compute_min_DCF(scores, labels, pi, 1, 1)
    return min_dcf, scores, labels
        
    
    
