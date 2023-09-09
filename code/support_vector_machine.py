
import numpy
import scipy
import validate
import matplotlib.pyplot as plt


def mrow(x):
    return numpy.reshape(x, (1,x.shape[0]))

def mcol(x):
    return numpy.reshape(x, (x.shape[0],1))
"""
----------------------------------------------------------------------------------------------------------------------
-----------------------------------------------LINEAR SVM-------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------
"""
#we are considering a binary problem, we're supposing that LTR is label 0 or 1
#C and K are hyperparameters. K is supposed to be alwasy = 1
def train_SVM_linear(DTR, LTR, C, piT, rebalance, K=1):
    
    #first we simulate the bias by extending our data with ones, this allow us to simulate the effect of a bias
    DTREXT=numpy.vstack([DTR, numpy.ones((1,DTR.shape[1]))])
    #DTREXt is the original data matrix  wich contains the original training data x1...xn. Each x has an added feature equals to K=1 appendend
    
    #compute Z to be used to do compute: zi(wTx+b)
    Z=numpy.zeros(LTR.shape)
    Z[LTR==1] = 1
    Z[LTR==0] =-1
    
    #compute matrix H which is inside the dual formula
    #Hij=zizjxiTxj 
    G=numpy.dot(DTREXT.T,DTREXT)
    H=mcol(Z) * mrow(Z) * G 
    
    def compute_JDual_and_gradient(alpha):
        Ha = numpy.dot(H,mcol(alpha))
        aHa = numpy.dot(mrow(alpha),Ha)
        a1=alpha.sum()
        JDual = -0.5 * aHa.ravel() +a1
        gradient = -Ha.ravel() + numpy.ones(alpha.size)
        return JDual, gradient
    
    def compute_LDual_and_gradient(alpha): #actually we'll minimize L(loss) rather than maxize J
        Ldual, grad = compute_JDual_and_gradient(alpha)
        return -Ldual, -grad
        
    if rebalance:
        DTR0 = DTR[:, LTR==0]
        DTR1 = DTR[:, LTR==1]
        pi_emp_F = (DTR0.shape[1] / DTR.shape[1])
        pi_emp_T = (DTR1.shape[1] / DTR.shape[1])
        CT = C * piT/pi_emp_T 
        CF = C * (1-piT)/pi_emp_F
        
        #bounds: a list of DTR.shape[1] elements. Each element is a tuple (0,CT) or (0,CF)
        bounds = [(0,1)]*DTR.shape[1] #Initialize the array with random values 
        for i in range(DTR.shape[1]):
            if LTR[i]==0:
                bounds[i]=(0,CF)
            else:
                bounds[i]=(0,CT)
    else:#no re-balancing
        bounds = [(0,C)]*DTR.shape[1]
        
    alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
                                                       compute_LDual_and_gradient, #function to minimze
                                                       numpy.zeros(DTR.shape[1]), #initiazilation affect only number of iterations to get to the minimum, but since the problem is convex the solution will be reached with any initial value 
                                                       bounds = bounds,
                                                       factr=1.0,
                                                       maxiter=100000,
                                                       maxfun= 100000, #these last 3 parameters define how good will be the computation  
                                                      )
    
    wStar = numpy.dot(DTREXT, mcol(alphaStar)*mcol(Z))

    return wStar, alphaStar

K_fold=5

def compute_score_linear(DTE,DTR,LTR, Options):
    if Options['C']== None:
        Options['C'] = 0
    if Options['piT'] == None:
        Options['piT']=0.5
    if Options['rebalance'] == None:
        Options['rebalance']=True

    w,_alfa= train_SVM_linear(DTR, LTR, Options['C'], Options['piT'], Options['rebalance'])
    DTE_EXT=numpy.vstack([DTE, numpy.ones((1,DTE.shape[1]))])
    score = numpy.dot(w.T,DTE_EXT)
    return score.ravel()
    
def plot_linear_minDCF_wrt_C(DTR,LTR,gaussianize, DEV=None, LEV=None, evaluation=False):
    print('Linear SVM: computation for plotting min_cdf wrt C started...')
    min_DCFs=[]
    for pi in [0.1, 0.5, 0.9]:
        C_array = numpy.logspace(-5,5, num = 30)
        for C in C_array:
                Options= {'C': C,
                          'piT':0.5,
                          'rebalance':True}
                min_dcf_kfold = validate.kfold(DTR, LTR, K_fold, pi, compute_score_linear, Options )[0] 
                min_DCFs.append(min_dcf_kfold)
                print ("Linear SVM min_dcf for pi=%f -C=%f - results min_dcf=%f "%(pi,C,min_dcf_kfold))
    min_DCFs_p0 = min_DCFs[0:30] #min_DCF results with prior = 0.1
    min_DCFs_p1 = min_DCFs[30:60] #min_DCF results with prior = 0.5
    min_DCFs_p2 = min_DCFs[60:90] #min_DCF results with prior = 0.9
    
    C_array = numpy.logspace(-5,5, num = 30)
   
    if evaluation==False: #plot only result for validation set
        plt.figure()
        plt.plot(C_array, min_DCFs_p0, label='prior=0.1')
        plt.plot(C_array, min_DCFs_p1, label='prior=0.5')
        plt.plot(C_array, min_DCFs_p2, label='prior=0.9')
        plt.legend()
        plt.semilogx()
        plt.xlabel("C")
        plt.ylabel("min_DCF")
        if gaussianize:
            plt.savefig("./images/min_DCF_C_linearSVM_gaussianized.png")
        else:
            plt.savefig("./images/min_DCF_C_linearSVM_raw.png")
        plt.show()
    
    else: #compare plot of validation and evaluation set
        min_DCFs=[]
        for pi in [0.1, 0.5, 0.9]:
            C_array = numpy.logspace(-5,5, num = 30)
            for C in C_array:
                    Options= {'C': C,
                              'piT':0.5,
                              'rebalance':True}
                    scores_linear_svm = compute_score_linear(DEV, DTR, LTR, Options)
                    min_DCF_ev = validate.compute_min_DCF(scores_linear_svm, LEV, pi, 1, 1)
                    min_DCFs.append(min_DCF_ev)
                    print ("computed min_dcf for pi=%f -C=%f - results min_dcf=%f "%(pi,C,min_DCF_ev))
        min_DCFs_p0_ev = min_DCFs[0:30] #min_DCF results with prior = 0.1
        min_DCFs_p1_ev = min_DCFs[30:60] #min_DCF results with prior = 0.5
        min_DCFs_p2_ev = min_DCFs[60:90] #min_DCF results with prior = 0.9
        
        C_array = numpy.logspace(-5,5, num = 30)
        
        
        plt.figure()
        plt.plot(C_array, min_DCFs_p0, '--b',  label='prior=0.1-val')
        plt.plot(C_array, min_DCFs_p1, '--r', label='prior=0.5-val')
        plt.plot(C_array, min_DCFs_p2, '--g', label='prior=0.9-val')
        plt.plot(C_array, min_DCFs_p0_ev, 'b', label='prior=0.1-eval')
        plt.plot(C_array, min_DCFs_p1_ev, 'r', label='prior=0.5-eval')
        plt.plot(C_array, min_DCFs_p2_ev, 'g', label='prior=0.9-eval')
        plt.semilogx()
        plt.xlabel("C")
        plt.ylabel("min_DCF")
        plt.legend()
        if gaussianize:
            plt.savefig("./images/min_DCF_C_linearSVM_evaluation_gaussianized.png")
        else:
            plt.savefig("./images/min_DCF_C_linearSVM_evaluation_raw.png")
        plt.show()
        
    #return min_DCFs

"""
----------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------RBF SVM ------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------
"""

def train_SVM_RBF(DTR, LTR, C, piT, gamma, rebalance, K=1):
    Z=numpy.zeros(LTR.shape)
    Z[LTR==1] = 1
    Z[LTR==0] =-1
    
    Dist= mcol((DTR**2).sum(0)) +mrow((DTR**2).sum(0)) - 2 * numpy.dot(DTR.T,DTR)
    kernel= numpy.exp(-gamma* Dist) + K #K account for byas term
    H=mcol(Z) *mrow(Z) * kernel
    
    def compute_JDual_and_gradient(alpha):
        Ha = numpy.dot(H,mcol(alpha))
        aHa = numpy.dot(mrow(alpha),Ha)
        a1=alpha.sum()
        JDual = -0.5 * aHa.ravel() +a1
        gradient = -Ha.ravel() + numpy.ones(alpha.size)
        return JDual, gradient
    
    def compute_LDual_and_gradient(alpha): #actually we'll minimize L(loss) rather than maxize J
        Ldual, grad = compute_JDual_and_gradient(alpha)
        return -Ldual, -grad
    
    
    if rebalance:
        DTR0 = DTR[:, LTR==0]
        DTR1 = DTR[:, LTR==1]
        pi_emp_F = (DTR0.shape[1] / DTR.shape[1])
        pi_emp_T = (DTR1.shape[1] / DTR.shape[1])
        CT = C * piT/pi_emp_T 
        CF = C * (1-piT)/pi_emp_F
        
        #bounds: a list of DTR.shape[1] elements. Each element is a tuple (0,CT) or (0,CF)
        bounds = [(0,1)]*DTR.shape[1] #Initialize the array with random values 
        for i in range(DTR.shape[1]):
            if LTR[i]==0:
                bounds[i]=(0,CF)
            else:
                bounds[i]=(0,CT)
    else:#no re-balancing
        bounds = [(0,C)]*DTR.shape[1]
        
    alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
                                                       compute_LDual_and_gradient, #function to minimze
                                                       numpy.zeros(DTR.shape[1]), #initiazilation affect only number of iterations to get to the minimum, but since the problem is convex the solution will be reached with any initial value 
                                                       bounds = bounds,
                                                       factr=1.0,
                                                       maxiter=100000,
                                                       maxfun= 100000, #these last 3 parameters define how good will be the computation  
                                                      )
    
    return alphaStar,Z



def compute_score_RBF(DTE,DTR,LTR, Options, K=1):
    if Options['C']== None:
        Options['C'] = 0
    if Options['piT'] == None:
        Options['piT']=0.5
    if Options['gamma']==None:
        Options['gamma']=0.1
    if Options['rebalance']==None:
        Options['rebalance']=True
    Dist= mcol((DTR**2).sum(0)) +mrow((DTE**2).sum(0)) - 2 * numpy.dot(DTR.T,DTE)
    kernel= numpy.exp(-Options['gamma']* Dist) + (K**2) #K account for byas term
    alphaStar, Z= train_SVM_RBF(DTR, LTR, Options['C'], Options['piT'] ,Options['gamma'],Options['rebalance'], K)
    score= numpy.dot(alphaStar * Z, kernel)
    return score.ravel()
        
   
def plot_RBF_minDCF_wrt_C(DTR,LTR,gaussianize, DEV=None, LEV=None, evaluation=False):
    print('RBF SVM: computation for plotting min_cdf wrt C started...')
    min_DCFs=[]
    pi=0.5
    gamma_array= [0.0001, 0.001, 0.01, 0.1]    
    for gamma in gamma_array:
        C_array = numpy.logspace(-5,5, num = 20)
        for C in C_array:
                Options= {'C': C,
                          'piT':0.5,
                          'gamma':gamma,
                          'rebalance':True}
                min_dcf_kfold = validate.kfold(DTR, LTR, K_fold, pi, compute_score_RBF, Options )[0] 
                min_DCFs.append(min_dcf_kfold)
                print ("computed min_dcf for pi=%f -gamma=%f -C=%f - results min_dcf=%f "%(pi, gamma, C,min_dcf_kfold))
        print (min_DCFs)
    min_DCFs_g0 = min_DCFs[0:20] #min_DCF results with gamma = 0.0001
    min_DCFs_g1 = min_DCFs[20:40] #min_DCF results with gamma = 0.001
    min_DCFs_g2 = min_DCFs[40:60] #min_DCF results with gamma = 0.01
    min_DCFs_g3 = min_DCFs[60:80] #min_DCF results with gamma = 0.1
    
    C_array = numpy.logspace(-5,5, num = 20)

    if evaluation == False:
        plt.figure()
        plt.plot(C_array, min_DCFs_g0, label='gamma=0.0001')
        plt.plot(C_array, min_DCFs_g1, label='gamma=0.001')
        plt.plot(C_array, min_DCFs_g2, label='gamma=0.01')
        plt.plot(C_array, min_DCFs_g3, label='gamma=0.1')
        plt.legend()
        plt.tight_layout() # TBR: Use with non-default font size to keep axis label inside the figure
        plt.semilogx()
        plt.xlabel("C")
        plt.ylabel("min_DCF")
        if gaussianize:
            plt.savefig("./images/min_DCF_C_RBF_SVM_zscore.png")
        else:
            plt.savefig("./images/min_DCF_C_RBF_SVM_.png")#################################################################################
        plt.show()
        
    else:#compare validation and evaluation
        min_DCFs=[]
        pi=0.5
        gamma_array= [0.0001, 0.001, 0.01, 0.1]
        for gamma in gamma_array:
            C_array = numpy.logspace(-5,5, num = 20)
            for C in C_array:
                    Options= {'C': C,
                              'piT':0.5,
                              'gamma':gamma,
                              'rebalance':True}
                    scores_rbf_svm = compute_score_RBF(DEV, DTR, LTR, Options)
                    min_DCF_ev =validate.compute_min_DCF(scores_rbf_svm, LEV, pi, 1, 1)
                    min_DCFs.append(min_DCF_ev)
                    print ("computed min_dcf for pi=%f -gamma=%f -C=%f - results min_dcf=%f "%(pi, gamma, C,min_DCF_ev))
        min_DCFs_g0_ev = min_DCFs[0:20] #min_DCF results with gamma = 0.0001
        min_DCFs_g1_ev = min_DCFs[20:40] #min_DCF results with gamma = 0.001
        min_DCFs_g2_ev = min_DCFs[40:60] #min_DCF results with gamma = 0.01
        min_DCFs_g3_ev = min_DCFs[60:80] #min_DCF results with gamma = 0.1
    
        
        plt.figure()
        plt.plot(C_array, min_DCFs_g0, '--b', label='γ=0.0001-val')
        plt.plot(C_array, min_DCFs_g1, '--g', label='γ=0.001-val')
        plt.plot(C_array, min_DCFs_g2, '--r', label='γ=0.01-val')
        plt.plot(C_array, min_DCFs_g3, '--y', label='γ=0.1-val')
        plt.plot(C_array, min_DCFs_g0_ev, 'b', label='γ=0.0001-eval')
        plt.plot(C_array, min_DCFs_g1_ev, 'g', label='γ=0.001-eval')
        plt.plot(C_array, min_DCFs_g2_ev, 'r', label='γ=0.01-eval')
        plt.plot(C_array, min_DCFs_g3_ev, 'y', label='γ=0.1-eval')
        plt.legend() #control if it has to be removed
        plt.tight_layout() 
        plt.semilogx()
        plt.xlabel("C")
        plt.ylabel("min_DCF")
        if gaussianize:
            plt.savefig("./images/min_DCF_C_RBF_SVM_eval_gaussianized.png")
        else:
            plt.savefig("./images/min_DCF_C_RBF_SVM__eval_raw.png")
        plt.show()
    
    return min_DCFs

"""
----------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------QUADRATRIC SVM------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------
"""

def train_SVM_quadratic(DTR, LTR, C, piT, rebalance, K=1):
    Z=numpy.zeros(LTR.shape)
    Z[LTR==1] = 1
    Z[LTR==0] =-1
    
    kernel= ((numpy.dot(DTR.T,DTR)+1)**2)+K #K account for byas term
    H=mcol(Z) *mrow(Z) * kernel
    
    def compute_JDual_and_gradient(alpha):
        Ha = numpy.dot(H,mcol(alpha))
        aHa = numpy.dot(mrow(alpha),Ha)
        a1=alpha.sum()
        JDual = -0.5 * aHa.ravel() +a1
        gradient = -Ha.ravel() + numpy.ones(alpha.size)
        return JDual, gradient
    
    def compute_LDual_and_gradient(alpha): #actually we'll minimize L(loss) rather than maxize J
        Ldual, grad = compute_JDual_and_gradient(alpha)
        return -Ldual, -grad

    if rebalance:
        DTR0 = DTR[:, LTR==0]
        DTR1 = DTR[:, LTR==1]
        pi_emp_F = (DTR0.shape[1] / DTR.shape[1])
        pi_emp_T = (DTR1.shape[1] / DTR.shape[1])
        CT = C * piT/pi_emp_T 
        CF = C * (1-piT)/pi_emp_F
        
        #bounds: a list of DTR.shape[1] elements. Each element is a tuple (0,CT) or (0,CF)
        bounds = [(0,1)]*DTR.shape[1] #Initialize the array with random values 
        for i in range(DTR.shape[1]):
            if LTR[i]==0:
                bounds[i]=(0,CF)
            else:
                bounds[i]=(0,CT)
    else:#no re-balancing
        bounds = [(0,C)]*DTR.shape[1]
            
    alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
                                                       compute_LDual_and_gradient, #function to minimze
                                                       numpy.zeros(DTR.shape[1]), #initiazilation affect only number of iterations to get to the minimum, but since the problem is convex the solution will be reached with any initial value 
                                                       bounds = bounds,
                                                       factr=1.0,
                                                       maxiter=100000,
                                                       maxfun= 100000, #these last 3 parameters define how good will be the computation  
                                                      )

    return  alphaStar, Z

K_fold=5
def compute_score_quadratic(DTE,DTR,LTR, Options, c=1, d=2, K=1):
    if Options['C']== None:
        Options['C'] = 0
    if Options['piT'] == None:
        Options['piT']=0.5
    if Options['rebalance']==None:
        Options['rebalance']=True
    kernel= ((numpy.dot(DTR.T,DTE)+c)**d)+K #K account for byas term
    alphaStar, Z= train_SVM_quadratic(DTR, LTR, Options['C'], Options['piT'], Options['rebalance'] )
    score= numpy.dot(alphaStar * Z, kernel)
    return score.ravel()
    
def plot_quadratic_minDCF_wrt_C(DTR,LTR,gaussianize, DEV=None, LEV=None, evaluation=False):
    print('Quadratic SVM: computation for plotting min_cdf wrt C started...')
    min_DCFs=[]
    for pi in [0.1, 0.5, 0.9]:
        C_array = numpy.logspace(-5,5, num = 30)
        for C in C_array:
                Options= {'C': C,
                          'piT':0.5,
                          'rebalance':True}
                min_dcf_kfold = validate.kfold(DTR, LTR, K_fold, pi, compute_score_quadratic, Options )[0] 
                min_DCFs.append(min_dcf_kfold)
                print ("computed min_dcf for pi=%f -C=%f - results min_dcf=%f "%(pi,C,min_dcf_kfold))
    min_DCFs_p0 = min_DCFs[0:30] #min_DCF results with prior = 0.1
    min_DCFs_p1 = min_DCFs[30:60] #min_DCF results with prior = 0.5
    min_DCFs_p2 = min_DCFs[60:90] #min_DCF results with prior = 0.9
    
    C_array = numpy.logspace(-5,5, num = 30)
   
    
    if evaluation == False:
        #setup visualization font
        plt.rc('font', size=16)
        plt.rc('xtick', labelsize=16)
        plt.rc('ytick', labelsize=16)
        plt.figure()
        plt.plot(C_array, min_DCFs_p0, label='prior=0.1')
        plt.plot(C_array, min_DCFs_p1, label='prior=0.5')
        plt.plot(C_array, min_DCFs_p2, label='prior=0.9')   
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        plt.semilogx()
        plt.xlabel("C")
        plt.ylabel("min_DCF")
        if gaussianize:
            plt.savefig("./images/min_DCF_C_QuadraticSVM_gaussianized.png")
        else:
            plt.savefig("./images/min_DCF_C_QuadraticSVM_raw.png")
        plt.show()
        
        
    else:#compare evaluation and validation
        print("Evaluation samples")
        min_DCFs=[]
        for pi in [0.1, 0.5, 0.9]:
            C_array = numpy.logspace(-5,5, num = 30)
            for C in C_array:
                    Options= {'C': C,
                              'piT':0.5,
                              'rebalance':True}
                    scores_svm_quad = compute_score_quadratic(DEV, DTR, LTR, Options)
                    min_DCF_ev = validate.compute_min_DCF(scores_svm_quad, LEV, pi, 1, 1)
                    min_DCFs.append(min_DCF_ev)
                    print ("computed min_dcf for pi=%f -C=%f - results min_dcf=%f "%(pi,C,min_DCF_ev))
            print(min_DCFs)
        min_DCFs_p0_eval = min_DCFs[0:30] #min_DCF results with prior = 0.1
        min_DCFs_p1_eval = min_DCFs[30:60] #min_DCF results with prior = 0.5
        min_DCFs_p2_eval = min_DCFs[60:90] #min_DCF results with prior = 0.9
        
       
        plt.figure()
        plt.plot(C_array, min_DCFs_p0, '--b', label='prior=0.1-val')
        plt.plot(C_array, min_DCFs_p1, '--r', label='prior=0.5-val')
        plt.plot(C_array, min_DCFs_p2, '--g', label='prior=0.9-val')   
        plt.plot(C_array, min_DCFs_p0_eval, 'b', label='prior=0.1-eval')
        plt.plot(C_array, min_DCFs_p1_eval, 'r', label='prior=0.5-eval')
        plt.plot(C_array, min_DCFs_p2_eval, 'g', label='prior=0.9-eval')  
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        plt.semilogx()
        plt.xlabel("C")
        plt.ylabel("min_DCF")
        if gaussianize:
            plt.savefig("./images/min_DCF_C_QuadraticSVM_eval_gaussianized.png")
        else:
            plt.savefig("./images/min_DCF_C_QuadraticSVM_eval_raw.png")
        plt.show()
        
    return min_DCFs
