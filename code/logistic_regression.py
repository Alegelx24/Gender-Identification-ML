import numpy
import scipy
import validate
import matplotlib.pyplot as plt

def mcol(v):
    return v.reshape((v.size, 1))

def mrow(v):
    return v.reshape((1, v.size))


K=5

#LINEAR LOG REG

def logreg_obj_wrap(DTR, LTR, l, piT): #l is lamda (used for regularization)
    #compute the labels Z
    Z= LTR * 2.0 -1.0
    M=DTR.shape[0]
    def logreg_obj(v):
        #v is a vector that contains [w,b]
        #extract b and w
        w = v [0:M]
        b = v[-1]
        
      
        DTR0 = DTR[:, LTR==0]
        DTR1 = DTR[:, LTR==1]
        Z0 = Z[LTR==0]
        Z1 = Z[LTR==1]
        
        S1=numpy.dot(w.T, DTR1)+b
        S0=numpy.dot(w.T, DTR0)+b #S= score= exponent of e = wTxi +b
        cxe=numpy.logaddexp(0,-S1*Z1).mean()* piT #cxe is the cross entropy = log [e^0 + e^(-sz)]
        cxe= cxe + numpy.logaddexp(0,-S0*Z0).mean()* (1-piT)
        return cxe+0.5*l*numpy.linalg.norm(w)**2 #I add also the regularization term
    return logreg_obj

def train_log_reg(DTR, LTR, lambdaa, piT, ):
    logreg_objective=logreg_obj_wrap(DTR, LTR, lambdaa, piT)
    _v,j,_d = scipy.optimize.fmin_l_bfgs_b(logreg_objective, numpy.zeros(DTR.shape[0]+1) , approx_grad=True) #numpy.zeros(DTR.shape[0]+1) is the starting point. I am providing w and b equal to 0 as starting point
    #I can recover optimal w* and b* from _v:
    _w=_v[0:DTR.shape[0]]
    _b=_v[-1]
    return _w,_b

def compute_score(DTE,DTR,LTR, Options):
    if Options['lambdaa']== None:
        Options['lambdaa'] = 0
    
    if Options['piT'] == None:
        Options['piT']=0.5
    
    _w,_b= train_log_reg(DTR, LTR, Options['lambdaa'], Options['piT'])
    scores = numpy.dot(_w.T,DTE)+_b
    return scores

def plot_minDCF_wrt_lamda(DTR,LTR, gaussianize, DEV=None, LEV=None, evaluation=False):
    print("plot of min_DCF wrt Lambda Linear Log Reg started...")
    min_DCFs=[]
    for pi in [0.1, 0.5, 0.9]:
        lambdas = numpy.logspace(-6,6, num = 40)
        for l in lambdas:
                Options= {'lambdaa':l,
                          'piT':0.5}
                min_dcf_kfold = validate.kfold(DTR, LTR, K, pi, compute_score, Options )[0]
                min_DCFs.append(min_dcf_kfold)

        min_DCFs_p0 = min_DCFs[0:40] #min_DCF results with prior = 0.1
        min_DCFs_p1 = min_DCFs[40:80] #min_DCF results with prior = 0.5
        min_DCFs_p2 = min_DCFs[80:120] #min_DCF results with prior = 0.9

    if evaluation==False: #plot onlly result for validation set
        plt.figure()
        plt.plot(lambdas, min_DCFs_p0, label='prior=0.1')
        plt.plot(lambdas, min_DCFs_p1, label='prior=0.5')
        plt.plot(lambdas, min_DCFs_p2, label='prior=0.9')
        plt.legend()
        plt.semilogx()
        plt.xlabel("λ")
        plt.ylabel("min_DCF")
        plt.savefig("./images/min_DCF_lambda_log_reg_raw.pdf")
        plt.show()    
    
    else: #compare plot of validation and evaluation set
        min_DCFs=[]
        for pi in [0.1, 0.5, 0.9]:
            lambdas = numpy.logspace(-6,6, num = 40)
            for l in lambdas:
                    Options= {'lambdaa':l,
                              'piT':0.5}
                    scores_linear_log_reg = compute_score(DEV, DTR, LTR, Options)
                    min_DCFs.append(validate.compute_min_DCF(scores_linear_log_reg, LEV, pi, 1, 1))
                    
        min_DCFs_p0_ev = min_DCFs[0:40] #min_DCF results with prior = 0.1
        min_DCFs_p1_ev = min_DCFs[40:80] #min_DCF results with prior = 0.5
        min_DCFs_p2_ev = min_DCFs[80:120] #min_DCF results with prior = 0.9
        plt.figure()
        plt.plot(lambdas, min_DCFs_p0, '--b',  label='prior=0.1-val')
        plt.plot(lambdas, min_DCFs_p1, '--r', label='prior=0.5-val')
        plt.plot(lambdas, min_DCFs_p2, '--g', label='prior=0.9-val')
        plt.plot(lambdas, min_DCFs_p0_ev, 'b', label='prior=0.1-eval')
        plt.plot(lambdas, min_DCFs_p1_ev, 'r', label='prior=0.5-eval')
        plt.plot(lambdas, min_DCFs_p2_ev, 'g', label='prior=0.9-eval')
        plt.semilogx()
        plt.xlabel("λ")
        plt.ylabel("min_DCF")
        plt.legend()
        if gaussianize:
            plt.savefig("./images/eval/min_DCF_lambda_log_reg_ev_val_gaussianized.pdf")
            plt.show()

        else:
            plt.savefig("./images/eval/min_DCF_lambda_log_reg_ev_val_raw.pdf")
            plt.show()

    return min_DCFs



#QUADRATIC LOG REG  

def compute_score_quadratic(DTE,DTR,LTR, Options):

    if Options['lambdaa']== None:
        Options['lambdaa'] = 0
    if Options['piT'] == None:
        Options['piT']=0.5
    
    phi = numpy.zeros((DTR.shape[0]**2 + DTR.shape[0], DTR.shape[1]))
    for i in range(DTR.shape[1]):
        x = DTR [:,i]
        x = mcol(x)
        product = numpy.dot(x,x.T) 
        vec = product.reshape((-1, 1))
        phi_col = numpy.concatenate((vec,x)).ravel()
        phi [:,i] = phi_col
    DTR = phi
    
    phi = numpy.zeros((DTE.shape[0]**2 + DTE.shape[0], DTE.shape[1]))
    for i in range(DTE.shape[1]):
        x = DTE [:,i]
        x = mcol(x)
        product = numpy.dot(x,x.T)
        vec = product.reshape((-1, 1), order="F")
        phi_col = numpy.concatenate((vec,x)).ravel()
        phi [:,i] = phi_col
    DTE = phi

    _w,_b= train_log_reg(DTR, LTR, Options['lambdaa'], Options['piT'])
    scores = numpy.dot(_w.T,DTE)+_b
    return scores
    
def quadratic_plot_minDCF_wrt_lambda(DTR,LTR, gaussianize, DEV=None, LEV=None, evaluation=False):
    print ('Quadratic Logistic Regression: computation for plot minDCF wt lambda started...')
    min_DCFs=[]
    for pi in [0.1, 0.5, 0.9]:
        lambdas = numpy.logspace(-6,6, num = 10)
        for l in lambdas:
                Options= {'lambdaa':l,
                          'piT':0.5}
                min_dcf_kfold = validate.kfold(DTR, LTR, K, pi, compute_score_quadratic, Options )[0]
                print ("computed min_dcf for pi=%f -lambda=%f - results min_dcf=%f "%(pi, l,min_dcf_kfold))
                min_DCFs.append(min_dcf_kfold)

    min_DCFs_p0 = min_DCFs[0:10] #min_DCF results with prior = 0.1
    min_DCFs_p1 = min_DCFs[10:20] #min_DCF results with prior = 0.5
    min_DCFs_p2 = min_DCFs[20:30] #min_DCF results with prior = 0.9

    if evaluation==False: #plot onlly result for validation set
        plt.figure()
        plt.plot(lambdas, min_DCFs_p0, label='prior=0.1')
        plt.plot(lambdas, min_DCFs_p1, label='prior=0.5')
        plt.plot(lambdas, min_DCFs_p2, label='prior=0.9')
        plt.legend()
        plt.semilogx()
        plt.xlabel("λ")
        plt.ylabel("min_DCF")
        if gaussianize:
            plt.savefig("./images/min_DCF_lamda_quadratic_log_reg_gaussianized.pdf")
        else:
            plt.savefig("./images/min_DCF_lamda_quadratic_log_reg_raw.pdf")
        plt.show()
        
    else: #compare plot of validation and evaluation set
        min_DCFs=[]
        for pi in [0.1, 0.5, 0.9]:
            lambdas = numpy.logspace(-6,6, num = 10)
            for l in lambdas:
                    Options= {'lambdaa':l,
                              'piT':0.5}
                    scores_quadraric_log_reg = compute_score_quadratic(DEV, DTR, LTR, Options)
                    min_DCFs.append(validate.compute_min_DCF(scores_quadraric_log_reg, LEV, pi, 1, 1))
                    
        min_DCFs_p0_ev = min_DCFs[0:10] #min_DCF results with prior = 0.1
        min_DCFs_p1_ev = min_DCFs[10:20] #min_DCF results with prior = 0.5
        min_DCFs_p2_ev = min_DCFs[20:30] #min_DCF results with prior = 0.9
        plt.figure()
        plt.plot(lambdas, min_DCFs_p0, '--b',  label='prior=0.1-val')
        plt.plot(lambdas, min_DCFs_p1, '--r', label='prior=0.5-val')
        plt.plot(lambdas, min_DCFs_p2, '--g', label='prior=0.9-val')
        plt.plot(lambdas, min_DCFs_p0_ev, 'b', label='prior=0.1-eval')
        plt.plot(lambdas, min_DCFs_p1_ev, 'r', label='prior=0.5-eval')
        plt.plot(lambdas, min_DCFs_p2_ev, 'g', label='prior=0.9-eval')
        plt.semilogx()
        plt.xlabel("λ")
        plt.ylabel("min_DCF")
        plt.legend()
        if gaussianize:
            plt.savefig("./images/eval/min_DCF_lamda_quadratic_log_reg_ev_val_gaussianized.pdf")
        else:
            plt.savefig("./images/eval/min_DCF_lamda_quadratic_log_reg_ev_val_raw.pdf")
        plt.show()
    
    return min_DCFs
