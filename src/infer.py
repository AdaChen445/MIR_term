from src.utils import ls, preprocess_wav, melspectrogram, to_numpy, plot_mel_transfer_infer, reconstruct_waveform
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch
import os
from src.transfer_models import *
from tqdm import tqdm 
import soundfile as sf
from src.params import sample_rate

# if src_id: 
#     ssim_recon = []
#     ssim_cyclic = []
# def infer(S):
#     """Takes in a standard sized spectrogram, returns timbre converted version"""
#     S = torch.from_numpy(S)
#     S = S.view(1, 1, img_height, img_width)
#     X = Variable(S.type(Tensor))
    
#     ret = {} # just stores inference output
    
#     mu, Z = encoder(X)
#     fake_X = G_trg(Z)
#     ret['fake'] = to_numpy(fake_X)
    
    
    # if src_id: 
    #     recon_X = G_src(Z)
    #     ret['recon'] = to_numpy(recon_X)
        
    #     mu_, Z_ = encoder(fake_X)
    #     cyclic_X = G_src(Z_)
    #     ret['cyclic'] = to_numpy(cyclic_X)
    
    # return ret
def audio_infer(wav, trg_id, src_id=None):
    model_path = r'./timbre_transfer_weight/'
    # encoder = torch.load(os.path.join(model_path, 'encoder_00.pt'))
    dim = 32
    channels = 1
    n_downsample = 2
    n_overlap = 4
    shared_dim = dim * 2 ** n_downsample
    root = r'./out/'
    os.makedirs(root+'/gen/', exist_ok=True)
    os.makedirs(root+'/ref/', exist_ok=True)
    encoder = Encoder(dim=dim, in_channels=channels, n_downsample=n_downsample)
    G_trg= Generator(dim=dim, out_channels=channels, n_upsample=n_downsample, shared_block=ResidualBlock(features=shared_dim))
    G_trg.load_state_dict(torch.load(os.path.join(model_path, 'G' + str(trg_id) + '_99.pth'),map_location ='cpu'))
    # G_src = torch.load(os.path.join(model_path, 'D3_00.pth'))
    encoder.load_state_dict(torch.load(os.path.join(model_path, 'encoder_99.pth'),map_location ='cpu'))
    img_width = 128
    img_height = 128
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    # Load audio and preprocess
    sample = preprocess_wav(wav)
    spect_src = melspectrogram(sample)
    
    #print(spect_src.max(), spect_src.min())

    spect_src = np.pad(spect_src, ((0,0),(img_width,img_width)), 'constant')  # padding for consistent overlap
    spect_trg = np.zeros(spect_src.shape)
    spect_recon = np.zeros(spect_src.shape)
    spect_cyclic = np.zeros(spect_src.shape)
    
    length = spect_src.shape[1]
    hop = img_width // n_overlap

    for i in tqdm(range(0, length, hop)):
        x = i + img_width

        # Get cropped spectro of right dims
        if x <= length:
            S = spect_src[:, i:x]
        else:  # pad sub on right if includes sub segments out of range
            S = spect_src[:, i:]
            S = np.pad(S, ((0,0),(x-length,0)), 'constant') 

        # ret = infer(S) # perform inference from trained model
        S = torch.from_numpy(S)
        S = S.view(1, 1, img_height, img_width)
        X = Variable(S.type(Tensor))
        
        ret = {} # just stores inference output
        
        mu, Z = encoder(X)
        fake_X = G_trg(Z)
        ret['fake'] = to_numpy(fake_X)
        T = ret['fake']
        # if src_id:
        #     R = ret['recon']
        #     C = ret['cyclic']

        # Add parts of target spectrogram with an average across overlapping segments    
        for j in range(0, img_width, hop):
            y = j + hop
            if i+y > length: break  # neglect sub segments out of range
                
            # select subsegments to consider for overlap
            t = T[:, j:y]
            # if src_id:
            #     r = R[:, j:y]
            #     c = C[:, j:y]
            
            # add average element
            spect_trg[:, i+j:i+y] += t/n_overlap
            # if src_id:
            #     spect_recon[:, i+j:i+y] += r/n_overlap
            #     spect_cyclic[:, i+j:i+y] += c/n_overlap


    # remove initial padding
    spect_src = spect_src[:, img_width:-img_width]
    spect_trg = spect_trg[:, img_width:-img_width]
    # if src_id:
    #     spect_recon = spect_recon[:, img_width:-img_width] 
    #     spect_cyclic = spect_cyclic[:, img_width:-img_width] 
                                                 
    # Compute and append SSIM
    # if src_id:
    #     ssim_recon.append(ssim(spect_src, spect_recon))
    #     ssim_cyclic.append(ssim(spect_src, spect_cyclic))

    # prepare file name for saving
    f = wav.split('/')[-1]
    wavname = f.split('.')[0]
    fname = 'G%s_%s' % (trg_id, wavname)

    # plot transfer if specified
    # if plot != -1:
    #     os.makedirs(root+'/plots/', exist_ok=True)
    #     plot_mel_transfer_infer(root+'/plots/%s.png' % fname, spect_src, spect_trg)  

    # reconstruct with Griffin Lim (takes a while, later feed this wav as input to vocoder)
    print('Reconstructing with Griffin Lim...')
    x = reconstruct_waveform(spect_trg)
    
    # sf.write(root+'/gen/%s_gen.wav'%fname, x, sample_rate)  # generated output
    sf.write(wav, x, sample_rate)  # generated output
    # sf.write(root+'/ref/%s_ref.wav'%fname, sample, sample_rate)  # input reference (for convenience)