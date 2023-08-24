
import dataset_util as util

k=5 #K-fold

  
def main():
    D,L = util.load_training_set('./data/Train.txt')
   # D= util.scale_ZNormalization(D)[0]
    #D_gaussianized= util.gaussianize_training(D)
    
    gaussianize= False 

    plot(D, L, gaussianize) #plot raw features before gaussianization
   # gaussianize= True
    #plot(D_gaussianized, L, gaussianize) #plot gaussianized features    
    



def plot(DTR, LTR, gaussianize):
    #save histograms of the distribution of all the features in '../Images' folder. E
    util.plot_histograms(DTR, LTR,gaussianize)
   # util.plot_scatters(DTR, LTR,gaussianize)
    util.heatmap_generator(DTR, LTR, "heatmap")


    # compute correlation of pearce for the features
    # util.pearce_correlation_map(DTR, LTR, gaussianize)

    
main()