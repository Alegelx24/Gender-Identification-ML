import numpy
import dataset_util as util


def mcol(v):
    return v.reshape((v.size,1))

def mrow(v):
    return v.reshape((1,v.size))

def full_compute_mean_covariance(D):
    mu=mcol(D.mean(1))
    C=numpy.dot(D-mu, (D-mu).T)/float(D.shape[1])
    return mu,C

def diag_compute_covariance(D, full_cov):
    C_diagonal = full_cov * numpy.identity(full_cov.shape[0])
    return C_diagonal

def tied_full_compute_covariance(D,L):
    C_tied=0
    for i in range(0,2):
        Dclass=D[:,L==i]
        _mu, C = full_compute_mean_covariance(Dclass)
        C_tied+=C * Dclass.shape[1]
    C_tied = C_tied/D.shape[1]
    return C_tied

def tied_diag_compute_covariance(D,L):
    C_tied=0
    for i in range(0,2):
        Dclass=D[:,L==i]
        _mu, C = full_compute_mean_covariance(Dclass)
        C = diag_compute_covariance(Dclass, C)
        C_tied+=C * Dclass.shape[1]
    C_tied = C_tied/D.shape[1]
    return C_tied

def logpdf_onesample(x, mu, C):
    P=numpy.linalg.inv(C)
    res= -0.5 * x.shape[0] * numpy.log(2*numpy.pi)
    res += -0.5*numpy.linalg.slogdet(C)[1] #slogdet return an array, the second element of the aray is the (natural) logarithm of the determinant of an array
    res += -0.5* numpy.dot( (x-mu).T,numpy.dot(P,(x-mu)))
    #res is a 1x1 matrix. I transform res in a 1-dimensional vector that contains only one value with the function ravel()
    return res.ravel()

def logpdf_GAU_ND(X,mu,C): #this function take as input a matrix of samples
    y=[logpdf_onesample(X[:, i:i+1], mu, C) for i in range(X.shape[1])]
    return numpy.array(y).ravel()
    #if i don't use ravel() i obtain a row-vector, while I want a 1-Dimensional vector

def train_gaussian_classifier(DTR,LTR, gaussian_type):
    D0 = DTR[:, LTR==0]
    D1 = DTR[:, LTR==1]
    #Insert kind of switch case which compure full, naive or tied
    mu0, C0 = full_compute_mean_covariance(D0)
    mu1, C1 = full_compute_mean_covariance(D1)
    if gaussian_type == 'diag':
        C0 = diag_compute_covariance(D0, C0)
        C1 = diag_compute_covariance(D1, C1)
    if gaussian_type == 'tied-full':
        C_tied = tied_full_compute_covariance(DTR, LTR)
        C0=C_tied
        C1=C_tied
    if gaussian_type == 'tied-diag':
        C_tied = tied_diag_compute_covariance(DTR, LTR)
        C0=C_tied
        C1=C_tied
    return mu0,C0,mu1,C1

def compute_score_full(DTE,DTR,LTR, Options):
    mu0,C0,mu1,C1= train_gaussian_classifier(DTR, LTR, 'full')
    log_density_c0 = logpdf_GAU_ND(DTE, mu0, C0)
    log_density_c1 = logpdf_GAU_ND(DTE, mu1, C1)
    score = log_density_c1 - log_density_c0
    return score

def compute_score_diag(DTE,DTR,LTR, Options):
    mu0,C0,mu1,C1= train_gaussian_classifier(DTR, LTR, 'diag')
    log_density_c0 = logpdf_GAU_ND(DTE, mu0, C0)
    log_density_c1 = logpdf_GAU_ND(DTE, mu1, C1)
    score = log_density_c1 - log_density_c0
    return score

def compute_score_tied_full(DTE,DTR,LTR, Options):
    mu0,C0,mu1,C1= train_gaussian_classifier(DTR, LTR, 'tied-full')
    log_density_c0 = logpdf_GAU_ND(DTE, mu0, C0)
    log_density_c1 = logpdf_GAU_ND(DTE, mu1, C1)
    score = log_density_c1 - log_density_c0
    return score

def compute_score_tied_diag(DTE,DTR,LTR, Options):
    mu0,C0,mu1,C1= train_gaussian_classifier(DTR, LTR, 'tied-diag')
    log_density_c0 = logpdf_GAU_ND(DTE, mu0, C0)
    log_density_c1 = logpdf_GAU_ND(DTE, mu1, C1)
    score = log_density_c1 - log_density_c0
    return score

    

    
    
    
