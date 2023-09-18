import numpy
import matplotlib.pyplot as plt
import scipy as sp

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


def center_dataset(DTR, DEV = None, normalize_ev=False): 
    mu=compute_mean(DTR)
    centered_DTR = (DTR-mu) 
    print('Z-Normalization done!')
     #normalize evaluation_set with mean and std of training set
    return centered_DTR, DEV


def plot_fraction_explained_variance_pca(DTR, LTR):
    DProjected, P = pca(DTR.shape[0] + 1, DTR)
    
    eigenvalues = numpy.linalg.eigvals(numpy.cov(DProjected))
    sorted_eigenvalues = numpy.sort(eigenvalues)[::-1]
    total_variance = numpy.sum(sorted_eigenvalues)
    explained_variance_ratio = numpy.cumsum(sorted_eigenvalues / total_variance)
    
    n_dimensions = DProjected.shape[0]
    plt.plot(range(1, n_dimensions + 1), explained_variance_ratio, marker='o')
    plt.xlabel('Number of dimensions')
    plt.ylabel('Fraction of explained variance')
    plt.savefig('./images/DatasetAnalysis/explained_variance_Pca.png')
    plt.close()


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
            plt.savefig('./images/DatasetAnalysis/hist/histogram_afterGaussianization_%d.png' % dIdx)
        else:
            plt.savefig('./images/DatasetAnalysis/hist/histogram_beforeGaussianization_zscore_%d.png' % dIdx)

        plt.show()
    
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
            plt.savefig('./images/DatasetAnalysis/scatter/scatter_%d_%d.png' % (dIdx1, dIdx2))
            plt.show()
        
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
        
        plt.savefig("./images/DatasetAnalysis/heatmaps/heatmap_%s.png" % (filename + "_" + titles[k]))

def pca(m, D):
    mu = compute_mean(D)
    DCentered = D - mu #center the data
    C=numpy.dot(DCentered,DCentered.transpose())/float(D.shape[1]) #compute emprical covariance matrix C
    _, U = numpy.linalg.eigh(C) #U contains the eigenvectors corresponding to eigenvalues of C in ascending order
    #I need to take the first m eigenvectors corresponding to the m largest eigenvalues
    P = U[:, ::-1][:, 0:m] #I invert the columns of U then I take the firsts m
    DProjected = numpy.dot(P.T, D)
    return DProjected, P



def vrow(col):
    return col.reshape((1,col.size))

def vcol(row):
    return row.reshape((row.size,1))

def createCenteredSWc(DC):      #for already centered data 
    C = 0
    for i in range(DC.shape[1]):
        C = C + numpy.dot(DC[:,i:i+1],(DC[:,i:i+1]).T)
    C = C/float(DC.shape[1])
    return C


def centerData(D):
    mu = D.mean(1)
    DC = D - vcol(mu)    #broadcasting applied
    return DC



def createSBSW(D,L):
    D0 = D[:,L==0]      
    D1 = D[:,L==1]

    DC0 = centerData(D0)
    DC1 = centerData(D1)

    SW0 = createCenteredSWc(DC0)
    SW1 = createCenteredSWc(DC1)
   
    centeredSamples = [DC0,DC1] 
    allSWc = [SW0,SW1]
    
    samples = [D0,D1]
    mu = vcol(D.mean(1))

    SB=0
    SW=0

    for x in range(2):
        m = vcol(samples[x].mean(1))
        SW = SW + (allSWc[x]*centeredSamples[x].shape[1]) 
        SB = SB + samples[x].shape[1] * numpy.dot((m-mu),(m-mu).T)                                                                 
        
    SB = SB/(float)(D.shape[1])
    SW = SW / (float)(D.shape[1])

    return SB,SW


def LDA1(D,L,m):

    SB, SW = createSBSW(D,L)        
    s,U = sp.linalg.eigh(SB,SW)  
    W = U[:,::-1][:,0:m]        

    return W




def LDA_impl(D,L,m) :
    W1 = LDA1(D,L,m)
    DW = numpy.dot(W1.T,D)     #D projection on sub-space W1
    return DW,W1



    

# 
    # DTEW = np.dot(W.T,DTE)
    #plotCross(DW,LTR,m)
    #plotSingle(DW, LTR, m)


    
# m is the number of dimensions
def plotLDA(D, L,m):


    DW,W = LDA_impl(D,L,m)
    D=DW

    D0 = D[:, L == 0]
    D1 = D[:, L == 1]
    for i in range(m):
            plt.figure()
            plt.xlabel("Feature " + str(i))
            plt.ylabel("Number of elements")
            plt.hist(D0[i, :],bins = 60, density = True, alpha = 0.4, label = 'Male' , color= 'blue', linewidth=1.0,
                     edgecolor='black')
            plt.hist(D1[i, :],bins = 60, density = True, alpha = 0.5, label = 'Female', color= 'pink', linewidth=1.0,
                     edgecolor='black')
            plt.legend()
            plt.tight_layout()
            plt.savefig("./images/DatasetAnalysis/LDA_plot.png" )
            plt.show()
