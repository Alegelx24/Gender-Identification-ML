import numpy
import pylab
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as statist



def mcol(v):
    return v.reshape((v.size, 1))

def mrow(v):
    return v.reshape((1, v.size))


def load_training_set(fname):
    #setup visualization font
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    
    DList = []
    labelsList = []
    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:12]
                attrs = mcol(numpy.array([float(i) for i in attrs]))
                label = line.split(',')[-1].strip()
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass
    DTR = numpy.hstack(DList)
    LTR =numpy.array(labelsList, dtype=numpy.int32)
    
    mean=compute_mean(DTR)
    std=compute_std(DTR)
    print ('The mean of the features for the entire training set is:')
    print(mean.ravel())
    print ('The standard deviation of the features for the entire training set is:')
    print(std.ravel())
    
    return DTR, LTR

def load_evaluation_set(fname):

    DList = []
    labelsList = []
    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:12]
                attrs = mcol(numpy.array([float(i) for i in attrs]))
                label = line.split(',')[-1].strip()
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass
    DEV = numpy.hstack(DList)
    LEV =numpy.array(labelsList, dtype=numpy.int32)

    return DEV, LEV


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1]) #idx contains N numbers from 0 to N (where is equals to number of training samples) in a random  order
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[:, idxTrain] 
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    
    return DTR, LTR, DTE, LTE

def compute_mean (D):
    mu = D.mean(1) #D is a matrix where each column is a vector, i need the mean for each feature
    return mu.reshape(mu.shape[0],1)

def compute_std (D):
    sigma = D.std(1) #D is a matrix where each column is a vector, i need the variance for each feature
    return sigma.reshape(sigma.shape[0],1)

def scale_ZNormalization(DTR, DEV = None, normalize_ev=False): 
    mu=compute_mean(DTR)
    sigma=compute_std(DTR)
    scaled_DTR = (DTR-mu) 
    scaled_DTR = scaled_DTR / sigma
    print('Z-Normalization done!')
    if normalize_ev:
        DEV=(DEV-mu)/sigma #normalize evaluation_set with mean and std of training set
    return scaled_DTR, DEV
    
def plot_histograms(D, L, gaussianize):

    D0 = D[:, L==0]
    D1 = D[:, L==1]

    feature_dict = {
        0: 'feature #1',
        1: 'feature #2',
        2: 'feature #3',
        3: 'feature #4',
        4: 'feature #5',
        5: 'feature #6',
        6: 'feature #7',
        7: 'feature #8',
        8: 'feature #9',
        9: 'feature #10',
        10: 'feature #11',
        11: 'feature #12'
    }


    for dIdx in range(12):
        plt.figure()
        plt.xlabel(feature_dict[dIdx])
        plt.hist(D0[dIdx, :], bins = 60, density = True, alpha = 0.4, label = 'Male' , color= 'blue', linewidth=1.0,
                     edgecolor='black')
        plt.hist(D1[dIdx, :], bins = 60, density = True, alpha = 0.5, label = 'Female', color= 'pink', linewidth=1.0,
                     edgecolor='black')
        
        plt.legend()
        plt.tight_layout() # TBR: Use with non-default font size to keep axis label inside the figure
        if gaussianize:
            plt.savefig('./images/DatasetAnalysis/hist/histogram_afterGaussianization_%d.pdf' % dIdx)
        else:
            plt.savefig('./images/DatasetAnalysis/hist/histogram_beforeGaussianization_%d.pdf' % dIdx)

    #plt.show()
    
def plot_scatters(D, L, gaussianize):
    
    D0 = D[:, L==0]
    D1 = D[:, L==1]

    feature_dict = {
        0: 'feature #1',
        1: 'feature #2',
        2: 'feature #3',
        3: 'feature #4',
        4: 'feature #5',
        5: 'feature #6',
        6: 'feature #7',
        7: 'feature #8',
        8: 'feature #9',
        9: 'feature #10',
        10: 'feature #11',
        11: 'feature #12'
    }

    for dIdx1 in range(12):
        for dIdx2 in range(12):
            if dIdx1 == dIdx2:
                continue
            plt.figure()
            plt.xlabel(feature_dict[dIdx1])
            plt.ylabel(feature_dict[dIdx2])
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label = 'Male' , color= 'blue', alpha=0.6)
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label = 'Female', color= 'pink', alpha=0.6)
            #TBR: The alpha blending value, between 0 (transparent) and 1 (opaque).
        
            plt.legend()
            plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
            plt.savefig('./images/DatasetAnalysis/scatter/scatter_%d_%d.pdf' % (dIdx1, dIdx2))
        #plt.show()
        
def heatmap_generator(D, L, filename):
    colors = ['YlOrBr', 'PuBuGn', 'RdPu']
    titles = {
        0: 'Whole dataset',
        1: 'Male',
        2: 'Female'
    }
    pearson_matrixes = {
        0: numpy.abs(numpy.corrcoef(D)),
        1: numpy.abs(numpy.corrcoef(D[:, L == 0])),
        2: numpy.abs(numpy.corrcoef(D[:, L == 1]))
    }
    
    for k in range(3):
        plt.figure()
        fig, ax = plt.subplots()
        
        # Create labels for the ticks on both axes
        labels = list(range(1, len(D) + 1))
        ax.set_xticks(numpy.arange(len(D)))
        ax.set_yticks(numpy.arange(len(D)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        
        for i in range(len(D)):
            for j in range(len(D)):
                matrix = pearson_matrixes[k]
                text = ax.text(j, i, numpy.round(matrix[i, j], 2),
                               ha="center", va="center", color="w", fontsize=8)
        
        ax.imshow(pearson_matrixes[k], cmap=colors[k], vmin=-1, vmax=1)
        
        plt.title(titles[k] + " - " + filename + " features")
        fig.tight_layout()
        
        plt.savefig("./images/DatasetAnalysis/heatmaps/heatmap_%s.pdf" % (filename + "_" + titles[k]))

    
    
    
    

def pca(m, D):
    mu = compute_mean(D)
    DCentered = D - mu #center the data
    C=numpy.dot(DCentered,DCentered.transpose())/float(D.shape[1]) #compute emprical covariance matrix C
    _, U = numpy.linalg.eigh(C) #U contains the eigenvectors corresponding to eigenvalues of C in ascending order
    #I need to take the first m eigenvectors corresponding to the m largest eigenvalues
    P = U[:, ::-1][:, 0:m] #I invert the columns of U then I take the firsts m
    DProjected = numpy.dot(P.T, D)
    return DProjected, P


    
    
    