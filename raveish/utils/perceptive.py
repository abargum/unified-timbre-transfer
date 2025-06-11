""" Code either inspired by or imported directly from: https://github.com/acids-ircam/variational-timbre.
    Esling, Chemla-Romeu-Santos, Bitton: GENERATIVE TIMBRE SPACES: REGULARIZING VARIATIONAL AUTO-ENCODERS WITH PERCEPTUAL METRICS, 2018"""

import numpy as np, os, pdb
from scipy.stats import norm
from .visualize_dimred import MDS
import sklearn.manifold as manifold

instruments = { 'Horn':0, 'Tenor-Trombone':1, 'Trumpet-C':2, 'Violin':3, 'Violoncello':4, 'Alto-Sax':5, 'Bassoon':6,
               'Clarinet-Bb':7, 'Flute':8, 'Oboe':9, '_length':10}

equivalenceInstruments = ['Clarinet-Bb', 'Alto-Sax', 'Trumpet-C', 'Violoncello', 
                          'Horn', 'Oboe', 'Flute',
                          'Bassoon', 'Tenor-Trombone', 'Violin']

def remove_instruments(instruments, ratings, to_remove):
    instruments = np.array(instruments)
    mask = ~np.isin(instruments, to_remove)
    updated_instruments = instruments[mask]
    updated_ratings = ratings[mask][:, mask]
    return updated_instruments, updated_ratings

def rename_instrument(instruments, ratings, old_name, new_name):
    instruments = np.array(instruments)
    mask = instruments == old_name
    instruments[mask] = new_name
    return instruments, ratings
    

def get_perceptual_centroids(mds_dims, timbre_path, covariance=True, timbreNormalize=True, timbreProcessing=True):
    if (timbreProcessing == True or (not os.path.isfile('timbre_' + str(mds_dims) + '.npy'))):
        fullTimbreData = np.load(timbre_path, allow_pickle=True).item()
        selectedInstruments = fullTimbreData['instruments']
        detailedMatrix = fullTimbreData['ratings']

        #update instruments used
        selectedInstruments, detailedMatrix = remove_instruments(selectedInstruments, detailedMatrix, 'Piano')
        selectedInstruments, detailedMatrix = remove_instruments(selectedInstruments, detailedMatrix, 'EnglishHorn')
        selectedInstruments, detailedMatrix = rename_instrument(selectedInstruments, detailedMatrix, 'FrenchHorn', 'Horn')
        
        # Final matrices
        nbIns = len(selectedInstruments)
        meanRatings = np.zeros((nbIns, nbIns))
        gaussMuRatings = np.zeros((nbIns, nbIns))
        gaussStdRatings = np.zeros((nbIns, nbIns))
        nbRatings = np.zeros((nbIns, nbIns))
        
        # Fit Gaussians for each of the sets of pairwise instruments ratings
        for i in range(nbIns):
            for j in range(i+1, nbIns):
                nbRatings[i, j] = detailedMatrix[i, j].size
                meanRatings[i, j] = np.mean(detailedMatrix[i, j])
                mu, std = norm.fit(detailedMatrix[i, j])
                gaussMuRatings[i, j] = mu
                gaussStdRatings[i, j] = std
                #print("%s vs. %s : mu = %.2f,  std = %.2f" % (selectedInstruments[i], selectedInstruments[j], mu, std))
        
        # Create square matrices
        meanRatings += meanRatings.T   
        gaussMuRatings += gaussMuRatings.T
        gaussStdRatings += gaussStdRatings.T
        meanRatings = (meanRatings - np.min(meanRatings)) / np.max(meanRatings)

        gaussMuRatings = (gaussMuRatings - np.min(gaussMuRatings)) / np.max(gaussMuRatings)
        gaussStdRatings = (gaussStdRatings - np.min(gaussStdRatings)) / np.max(gaussStdRatings)
        variance = np.mean(gaussStdRatings, axis=1)
        if (timbreNormalize):
            variance = ((variance - (np.min(variance)) + 0.01) / np.max(variance)) * 2
        
        seed = np.random.RandomState(seed=3)        
        mds = manifold.MDS(n_components=mds_dims, max_iter=3000, eps=1e-9, random_state=seed, dissimilarity="precomputed", n_jobs=1)
        position = mds.fit(gaussMuRatings).embedding_

        fullTimbreData = {'instruments':selectedInstruments, 
                          'ratings':detailedMatrix,
                          'gmean':gaussMuRatings,
                          'gstd':gaussStdRatings,
                          'pos':position,
                          'var':variance}
        np.save('timbre_' + str(mds_dims) + '.npy', fullTimbreData)
    else:
        # Retrieve final data structure
        fullTimbreData = np.load('timbre.npy').item()
        selectedInstruments = fullTimbreData['instruments']
        gaussMuRatings = fullTimbreData['gmean']
        gaussStdRatings = fullTimbreData['gstd']
        position = fullTimbreData['pos']
        variance = fullTimbreData['var']
        
    audioTimbreIDs = np.zeros(len(equivalenceInstruments)).astype('int')
    
    # Parse through the list of instruments
    for k, v in instruments.items():
        if (k != '_length'):
            audioTimbreIDs[v] = equivalenceInstruments.index(k)
    prior_mean = position[audioTimbreIDs]
    prior_std = np.ones((len(equivalenceInstruments), mds_dims))
    if (covariance == 1):
        prior_std = prior_std * variance[audioTimbreIDs, np.newaxis]
    prior_params = (prior_mean, prior_std)
    prior_gauss_params = (gaussMuRatings, gaussStdRatings)  

    #return prior_params, prior_gauss_params, fullTimbreData
    return fullTimbreData