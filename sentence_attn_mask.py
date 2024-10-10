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
from transformers import AutoTokenizer, GPT2Model
import torch
import torch.utils.data as data
from torch.nn import functional as F

# tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2",add_prefix_space=True)
# model = GPT2Model.from_pretrained("openai-community/gpt2")


def parse_arguments():
    """Read commandline arguments

    Returns:
        args (Namespace): input as well as default arguments
    """

    parser = argparse.ArgumentParser()

    # main args
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--head", type=int, required=True) 
    parser.add_argument("--context_length", type=int, required=True)
    # parser.add_argument("--model-size", type=str, required=True)
    # parser.add_argument("--data-dir", type=str, required=True)
    # parser.add_argument("--saving-dir", type=str, required=True)
    # parser.add_argument("--freeze-decoder", action="store_true")
    # parser.add_argument("--training-step", type=int, default="3000")
    # parser.add_argument("--z_score", action="store_true")
    # parser.add_argument("--make_position_embedding_zero", action="store_true")

    args = parser.parse_args()

    # Set parameters
    # args.data_dir = os.path.join(
    #     "/scratch/gpfs/kw1166/decoding-challenge/data/", args.data_dir
    # )
    # args.max_neural_len = 919  # length of max neural signal
    # # args.max_source_positions = 919  # length of encoder hidden states
    # args.max_source_positions = 460  # length of encoder hidden states
    # args.grid_elec_num = 64  # num of elec per grid

    # # load electrode grid
    # if args.grid == "all":
    #     args.grids = [1, 2, 3, 4]
    # elif args.grid == "6v":
    #     args.grids = [1, 2]
    # elif args.grid == "BA44":
    #     args.grids = [3, 4]
    # else:
    #     assert args.grid.isdigit()
    #     args.grids = int(args.grid)

    # # for the grids, load electrode idx
    # eleclist = []
    # for grid in args.grids:
    #     gridfile = glob.glob(os.path.join("data", "grids", f"grid-{grid}*.txt"))
    #     with open(gridfile[0]) as f:
    #         while line := f.readline():
    #             elec = line.rstrip()
    #             assert elec.isdigit()
    #             eleclist.append(int(elec))
    # args.eleclist = eleclist

    # args.features = args.feature.split("-")
    # args.feature_dim = len(args.grids) * len(args.features)
    # args.num_mel_bins = args.grid_elec_num * args.feature_dim

    # path='/scratch/gpfs/arnab/decoding_challenge/saved_models/'
    # args.saving_dir = path+ args.saving_dir
    # # args.saving_dir = os.path.join("models", args.saving_dir)
    # write_model_config(vars(args))

    # args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args


args = parse_arguments()

model_name='gpt2-xl'

batch_size=30
# layer=48
# head=13

layer=args.layer
head=args.head


start=4000
context=args.context_length



filename='attention_'+model_name+'_layer_'+str(layer)+'_head_'+str(head)+'_context_'+str(context)+'.pkl'

print(filename)

cache_dir='/scratch/gpfs/arnab/.cache'

def download_hf_model(
    model_name, model_class, tokenizer_class, cache_dir, local_files_only=True 
):
    """Download a Huggingface model from the model repository (cache)."""
    
    # if cache_dir is None:
    #     cache_dir = set_cache_dir()

    model = model_class.from_pretrained(
        model_name,
        output_hidden_states=True,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    tokenizer = tokenizer_class.from_pretrained(
        model_name,
        add_prefix_space=True,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    return model, tokenizer


    

model, tokenizer=download_hf_model(model_name, GPT2Model, AutoTokenizer, cache_dir, local_files_only=True)   
tokenizer.pad_token = tokenizer.eos_token  

#a=tokenizer.tokenize('rainforest', return_tensors="pt")
# breakpoint() 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)



### loading ecog, embeddings, sentences

path_ken1="/scratch/gpfs/arnab/sentence_encoding/embeddings/"
model_size='gpt2-xl'
path_ken=path_ken1+model_size+'/full/'
os.chdir(path_ken)

pickle_file = open("base_df.pkl", "rb")
objects = []

i=0

while True:
    print('i',i)
    try:

        objects.append(pickle.load(pickle_file))

    except EOFError:

        break

pickle_file.close()

a=objects[0]
df=pd.DataFrame(a)
df2=df.loc[~df.adjusted_onset.isnull() & (df.token_idx==0)] # and df.token_idx==1
index=df2.index

onsets=np.squeeze(df2.adjusted_onset.values)
words=df2.word.values
del a

# ##embedding
# path_ken1="/scratch/gpfs/arnab/sentence_encoding//embeddings/"
# model_size='gpt2-xl'
# path_ken=path_ken1+model_size+'/full/'
# path_emb=path_ken+'cnxt_0050/'
# os.chdir(path_emb)

# emb_file='layer_48.pkl'

# pickle_file = open(emb_file, "rb")
# objects = []

# i=0

# while True:
#     print('i',i)
#     try:

#         objects.append(pickle.load(pickle_file))

#     except EOFError:

#         break

# pickle_file.close()

# a=objects[0]

# df_emb=pd.DataFrame(a)
# df_bert=df_emb.iloc[index]
# w=(df_bert.embeddings.values)
# # type(w[0])
# # np.shape(w)
# emb=np.zeros((len(w),np.shape(w[0])[0]))
# for i in range(len(w)):
#     emb[i,:]=w[i]
    
# print(np.shape(emb))

# # path="/scratch/gpfs/arnab/Encoding/different_size/"
# # os.chdir(path)
# # filename=model_size+'.mat'
# # savemat(filename,{'onset':onset,'embeddings':emb})
# del emb

##sentences


sentences=[]
for i in range(start+context,len(words)):
# i=10
    a=words[i-context:i]
    
    s=[]
    s=(a[0].replace(" ", ""))
    for k in range(1,context):
        s=s+' '+(a[k].replace(" ", ""))

    sentences.append(s)
    
# sentence    
print(len(sentences))

# all_token=[]

# for k in range(len(sentences)):

#     inputs = tokenizer(sentences[k], return_tensors="pt")
#     e=inputs['input_ids']

#     all_token.append(e)

# def make_dataloader_from_input(windows, batch_size):
#     input_ids = torch.tensor(windows)
#     data_dl = data.DataLoader(input_ids, batch_size=batch_size, shuffle=False)
#     return data_dl


# s=make_dataloader_from_input(all_token, batch_size=100)


# df_sentence=pd.read_csv('/scratch/gpfs/arnab/sentence_encoding/all_sentence_podcast.csv')
# num_word=df_sentence['num_word']
# cumulative_words=np.cumsum(num_word)

input_data= data.DataLoader(sentences, batch_size=batch_size, shuffle=False)
attn=[]

with torch.no_grad():
        model = model.to(device)
        model.eval()

        for batch_idx, batch in enumerate(input_data):
            # if batch_idx % 10 == 0:
            print(f"Batch ID: {batch_idx}")

            # breakpoint() 

            # batch = batch.to(device)
            inputs = tokenizer(batch, return_tensors="pt",padding="longest")
            inputs = inputs.to(device)
            outputs = model(**inputs, output_attentions=True)
            # e1=inputs['input_ids']

            for k in range(inputs['input_ids'].shape[0]):

                a1=outputs.attentions[layer-1][k][head-1]
                max_len=a1.shape[0]
                # e=e[k]

                idx=[]
                
                tokens=tokenizer.tokenize(batch[k], return_tensors="pt")

                for ii in range(len(tokens)):

                    if tokens[ii][0]=='Ġ':
                        idx.append(1)

                    else:
                        idx.append(0)

                if len(idx)<max_len:

                    for qq in range(max_len-len(idx)):
                        idx.append(0)

                indices=np.where(np.asarray(idx)==0)[0]
                a1=a1.cpu().detach().numpy()
                # breakpoint() 
                a1=np.delete(a1,indices, axis=0)
                a1= np.delete(a1,indices, axis=1)                
                # a1=F.softmax(a1[context-2,:context-1],dim=0)
                a1=a1[context-2,:context-1]


                attn.append(a1)

                del a1
                del indices
                

            # breakpoint() 

# for k in range(len(sentences)):

#     if k%10==0:

#         print(k)

#     inputs = tokenizer(sentences[k], return_tensors="pt")
#     outputs = model(**inputs, output_attentions=True)

#     attn=outputs.attentions[layer-1][0][head-1]
#     attention.append(attn)

#     e=inputs['input_ids']

#     word=[]
#     idx=[]
#     for i in range(e.shape[1]):
#         word.append(tokenizer.decode(e[0][i]))

#     sentence_words.append(word)

#     tokens=tokenizer.tokenize(sentences[k], return_tensors="pt")

#     for ii in range(len(tokens)):

#         if tokens[ii][0]=='Ġ':
#             idx.append(1)

#         else:
#             idx.append(0)

#     token_idx.append(np.asarray(idx))

#     # breakpoint()
path1='/scratch/gpfs/arnab/sentence_encoding/results/gpt2-xl/' 
path=path1+'layer_'+str(layer)+'/context_'+str(context)+'/'


# path='/scratch/gpfs/arnab/sentence_encoding/results/gpt2-xl/layer_48/'    
os.chdir(path)
db = {}
db['attention_mask'] = attn
# db['sentence_words'] = sentence_words
# db['token_idx'] = token_idx

with open(filename, 'wb') as handle:
    pickle.dump(db, handle, protocol=pickle.HIGHEST_PROTOCOL)

# breakpoint()

# def load_pickle(file_path):
    
#     pickle_file = open(file_path, "rb")
#     objects = []

#     i=0

#     while True:
#         print('i',i)
#         try:

#             objects.append(pickle.load(pickle_file))

#         except EOFError:

#             break

#     pickle_file.close()

#     a=objects[0]
    
#     return a

# def load_label(filepath):

#     with open(filepath, "rb") as f:
#         full_labels = pickle.load(f)
#         #labels_df = pd.DataFrame(full_labels["labels"])
#         labels_df = pd.DataFrame(full_labels)

#     # labels_df["audio_onset"] = ((labels_df.onset + 3000) / 512)
#     # labels_df["audio_offset"] = ((labels_df.offset + 3000) / 512)

#     # labels_df = labels_df.dropna(subset=["audio_onset", "audio_offset"])
    

#     return labels_df

# pickle_file_path='attention.pickle'

# df=load_pickle(pickle_file_path)

# breakpoint() 
