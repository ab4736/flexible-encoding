# imports below 

import os
import glob
import pickle
import numpy as np
import pandas as pd  
import csv
import string
from scipy.io import savemat
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from functools import reduce 
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import argparse
import torch
from torch import nn
from einops import rearrange
import torch.utils.data as data
from torch.nn import functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# setting global variables below
batch_size=50 # number of samples per batch for training
num_words=500 # total number of words in the vocabulary
emb_dim=50 # dimensionality of word embeddings
num_lags=3 # number of time lags in electrode data
EPOCHS = 5 # how many times are all the data points iterated through


electrode=5  #which electrode are we considering
taking_words=False  # are we taking the actual words or the lags around the last word
lag_number=3 # how many previous lags/words to consider
lags=[0,50,100] # only valid if taking word is false and should equal to the lagn_number
train_num=3500 # 
subject=798 # which subject is being tested
min_num_words=5 # filter the other senctences
audio_set=True # if taking into account audio onset and offset



def load_pickle(file_path): # loads object from a pickle file
    
    # opens the file and creates an objects list to hold objects
    pickle_file = open(file_path, "rb")
    objects = []
    
    # loops through the file, and adds each object to the above list, returning first object
    while True:
        try:
            objects.append(pickle.load(pickle_file)) 
        except EOFError:
            break

    pickle_file.close()
    first = objects[0]
    return first

def get_elec_id(subject):  # gets the IDs for each of the electrodes

    # creates the dataframe of electrode label names for a given subject
    path = '/projects/HASSON/247/plotting/sig-elecs/20230510-tfs-sig-file/'
    sig_file = path+'tfs-sig-file-glove-'+ str(subject)+'-'+'comp'+'.csv'
    df_sig = pd.read_csv(sig_file) 

    # extracts electrode signal values from the dataframe
    elecs = df_sig.electrode.values 

    # creates a dataframe of the processed electrode data names
    path = '/scratch/gpfs/kw1166/247/247-pickling/results/tfs/'+str(subject)+'/pickles/'+str(subject)+'_electrode_names.pkl'
    pickle_file = load_pickle(path)
    df_name_match=pd.DataFrame(pickle_file)

    # creates a list of electrode_id's from the electrode names (for example, SG2 is encoded as 194)
    elec_id=[]
    for elec in elecs:
        elec_id.extend(df_name_match[df_name_match.electrode_name==elec].electrode_id.values) 

    return elec_id

def create_dataframe(subject): # creates the dataframe from the encoding data

    # reads the final electrode decoding data into dataframe df
    path = "/scratch/gpfs/arnab/247_data_read/decoding_df_" + str(subject) + "_final.csv"
    df = pd.read_csv(path)

    # loads the last word embeddings into data_embeds 
    data_embeds=load_pickle('/scratch/gpfs/arnab/247_data_read/last_word_embeddings.pkl')
    emb = data_embeds["embeddings"]
    all_onsets = data_embeds["all_onsets"]

    # adds the embeddings and all_onsets columns to df
    df["embeddings"] = emb
    df["all_onsets"] = all_onsets

    # drops duplicates in the onsets column
    df=df.drop_duplicates(subset=['onsets'])

    # filters the dataframe based on minimum word number and if it is corrupted
    df=df[df.num_words > min_num_words]
    df=df[df.corrupted==0]

    return df

def all_ecog(conv_name, subject): # returns all the ECOG data for a given conversation name and subject

    # gets the id's for each electrode and stores how many there are for the given conversation
    elec_id = get_elec_id(subject)
    elec_num = len(elec_id)

    # add the ecog data
    ecogs=[]
    path='/projects/HASSON/247/data/conversations-car/' + str(subject) + '/'
    for electrode in elec_id:
        filename = path+'/'+conv_name+'/preprocessed_allElec/'+conv_name+'_electrode_preprocess_file_'+str(electrode)+'.mat'
        e = loadmat(filename)['p1st'].squeeze().astype(np.float32)
        ecogs.append(e)
        
    # create a numpy array of ecog data
    ecogs = np.asarray(ecogs).T
    print(ecogs)

    return ecogs

def get_elec_data(subject, df, ecogs, conv_name, all_onsets, lags, elec_id, lag_number,taking_words=True): # extract onset signal data from the electrodes

    # set the window size as well as variables representing the time
    elec_num=len(elec_id)
    t = len(ecogs[:,0])   
    window_size=200
    half_window = round((window_size / 1000) * 512 / 2)

    # if we using words as separation, then set the onsets equal to the word onsets
    if taking_words:
        Y_data = np.zeros((elec_num, lag_number))    
        onsets = all_onsets[-lag_number:-1]
        onsets = np.append(onsets, all_onsets[-1])

    # if not, then use sliding window formula to set the onsets
    else: 
        onsets=[]
        Y_data= np.zeros((elec_num, len(lags)))    

        for i in lags:

            lag_amount = int(i/ 1000 * 512)

            onsets.append(np.minimum(
                t - half_window - 1,
                np.maximum(half_window + 1,
                            np.round(all_onsets[-1]) + lag_amount)))
    
    # converting the ondets into np array
    index_onsets=np.asarray(onsets)

    # for loop looping through each electrode
    for k in range(np.shape(ecogs)[1]):

        # initialize array to store data in window
        Y1 = np.zeros((len(onsets), 2 * half_window + 1))

        # get ecog signal data for the current electrode
        brain_signal = ecogs[:,k]
        
        # subtracting 1 from starts to account for 0-indexing, setting the window bounds
        starts = index_onsets - half_window - 1
        stops = index_onsets + half_window

        # loop through each sets of onsets, and extract windowed data
        for i, (start, stop) in enumerate(zip(starts, stops)):
            start = int(start)
            stop = int(stop)
            Y1[i, :] = brain_signal[start:stop].reshape(-1)

        # calculate the mean across time for each onset window and update electrode data accordingly 
        Y_data[k,:] = np.mean(Y1, axis = -1)

    return Y_data


def prepare_emb_electrode(df, ecogs, subject, elec_id): # preparing the electrode_data and embeddings

    # preparing the arrays holding the embeddings and onsets
    df = create_dataframe(subject)
    embeddings = df.embeddings.values
    all_onsets = df.all_onsets.values

    # create an object for conversation names
    conv_names = np.unique(df.conversation_name.values)

    # conditional on taking_words, create objects for lagged electrode data with dimensions (conv #, elec #, lag #)
    if taking_words:
        electrode_data = np.zeros((len(df.conversation_name.values),len(elec_id),lag_number)) 
    else:
        electrode_data = np.zeros((len(df.conversation_name.values),len(elec_id),len(lags))) 

    # declaring embeddings
    embeddings = []

    # loop through each conversation and update the dataframe, embeddings, onset, and ecog values
    for conv_name in conv_names:
        df_current = df[df.conversation_name==conv_name]
        embeddings_current = df_current.embeddings.values
        onsets_current = df_current.all_onsets.values
        ecogs_current = all_ecog(elec_id, conv_name,subject)
    
        for k in range(len(onsets_current)): 
            conv_name=df.conversation_name.values[k]
            onset = onsets_current[k]

            # if statement checking if there are enough data points in onset, and that the last onset occurs before end of ECOG recording
            if len(onset)>= lag_number and onset[-1] < len(ecogs_current[:,0]):

                # getting the electrode data, then adding it to the list of total data
                current_elec_data = get_elec_data(subject, df, ecogs_current, conv_name, x, lags, elec_id, lag_number, taking_words)
                electrode_data[p,:,:] = current_elec_data 
                embeddings.append(embeddings_current[k])
                p = p + 1
    
    # filtering the electrode data on the total number of proper data points
    electrode_data=electrode_data[:p,:,:]

    # changing embeddings to have emb_dim dimensions, and creating an extra axis 
    embeddings = np.asarray(embeddings)
    pca = PCA(n_components=emb_dim)
    embeddings = pca.fit_transform(embeddings)
    embeddings = np.expand_dims(embeddings, axis=1)

    # returning embeddings and electrode_data
    return embeddings, electrode_data

def prepare_train_test(df, ecogs, subject, elec_id, train_num, batch_size, electrode): # creating the train and test splits and loaders
    
    # using the above prepare_emb_electrode method for data
    embeddings, electrode_data = prepare_emb_electrode(df, ecogs, subject, elec_id)

    # getting data for a specific electrode
    elec_data = electrode_data[:,electrode,:]

    # expand the dimensions of the elec_data by 1
    elec_data = np.expand_dims(elec_data, axis=1)

    # create the train test split for X and y
    X_train=torch.from_numpy(embeddings[:train_num,:,:])
    y_train=torch.from_numpy(elec_data[:train_num,:,:])

    X_test=torch.from_numpy(embeddings[train_num:,:,:])
    y_test=torch.from_numpy(elec_data[train_num:,:,:])

    # create the train and test set and loaders from the data
    trainset = torch.utils.data.TensorDataset(X_train, y_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)

    testset = torch.utils.data.TensorDataset(X_train, y_train)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)

    return trainset, trainloader, testset, testloader

class flex_encoding(nn.Module):
    def __init__(self, emb_dim, num_lags):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, num_lags)
    
    def forward(self, x):
        
        y_pred= (self.fc1(x))        
        return y_pred,v
    

def bdot(a, b):  
    B = a.shape[0]
    S = a.shape[2]
    return torch.bmm(a.view(B, 1, S), b.view(B, S, 1)).reshape(-1)

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, y_pred, v, targets):
        mse_error = torch.square(targets-y_pred)
        loss=bdot(v, mse_error)
        return loss.mean()  
    





def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.
    
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for batch_idx, batch in enumerate(trainloader):
        # Every data instance is an input + label pair
        x,y=batch
    
        # Zero your gradients for every batch!
        optimizer.zero_grad()
    
        # Make predictions for this batch
        x = x.to(torch.float32)
        x = x.to(device)
        y = y.to(torch.float32)
        y = y.to(device)
        [output1,output2]=model(x)
    
        # Compute the loss and its gradients
        loss = loss_fn(output1,output2, y)
        loss.backward()
    
        # Adjust learning weights
        optimizer.step()
    
        # Gather data and report
        running_loss += loss.item()

    last_loss = running_loss / (batch_idx+1) # loss per batch
    print('epoch {} loss: {}'.format(epoch_index, last_loss))
            
    return last_loss

# below is the training loop for the model
for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(testloader):
            vinputs, vlabels = vdata
            vinputs = vinputs.to(torch.float32)
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(torch.float32)
            vlabels = vlabels.to(device)
            [output1,output2]=model(vinputs)
            voutputs = model(vinputs)
            vloss = loss_fn(output1,output2, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))


