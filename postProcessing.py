import numpy as np
import pdb

from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture as GMM
from sklearn.model_selection import KFold
from utils import fisher_vector, align
import sys, glob, argparse
import numpy as np
import math, cv2
from scipy.stats import multivariate_normal
import time
from sklearn import svm

N = 350
K_Series = {2}

def train(features, labels):
    X = features
    Y = labels;
    
    clf = svm.SVC(kernel='linear', class_weight='balanced')
    clf.fit(X,Y)
    return clf
    
def success_rate(classifier, features, labels):
    print("Applying the classifier...")
    X = features
    Y = labels
    print(classifier.predict(X))
    res = float(sum([a==b for a,b in zip(classifier.predict(X), Y)])) / len(Y)
    return res


def main():

    featureIn = np.load("RESULT/features_raw.npy") #(350,16,16,512)
    featureIn=np.reshape(featureIn,(350,256,-1)) #(350,256,512)
    labelIn = np.load("RESULT/corspdn_labels.npy")
    
    lbl_re = labelIn[::10]
    xx = featureIn

    
    for K in K_Series:
        shapeminus1 = 1025*K
        fv = []
        fv_re = np.ones((35,10,shapeminus1))
        
        for item in xx:
            gmm = GMM(n_components=K, covariance_type='diag',reg_covar=1e-6)
            gmm.fit(item)
            fv_ = fisher_vector(item, gmm)
            fv.append(fv_)
        fv = np.asarray(fv) #(350, 4100)
        
        for i in range(350):
            fv_re[int(i/10)][i%10] = fv[i]
        
        print(fv_re.shape)
        fv_re = np.reshape(fv_re,(35,shapeminus1*10))
        print(fv_re.shape)
        
        # Do some shuffling.
        state = np.random.get_state()
        np.random.shuffle(fv_re)
        np.random.set_state(state)
        np.random.shuffle(lbl_re)
        
        
        kf = KFold(n_splits=7)
        
        epoch = 0
        rate_sum = 0
        for Idx_train, Idx_test in kf.split(fv_re,lbl_re):
            epoch+=1
            fv_re_train = align(fv_re,Idx_train)
            lbl_re_train = align(lbl_re, Idx_train)
            fv_re_test = align(fv_re,Idx_test)
            lbl_re_test = align(lbl_re, Idx_test)
            classifier = train(fv_re_train, lbl_re_train)
            rate = success_rate(classifier, fv_re_test, lbl_re_test)
            rate_sum+=rate
            print("epoch = {}".format(epoch),"  K:",K,"  success rate:", rate)
        print("\n===average=acc=is={:.4f}".format(rate_sum/7.0))
        
        
if __name__ == '__main__':
    main()
    
    
    
    
    
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    