#!/usr/bin/python
#coding:utf-8
"""
動画からフレームと音声を画像データにして保存するスクリプト．

  $ python generate_data.py 入力の動画ファイル
"""
import os
import sys
import argparse
import numpy as np
import yaml
import moviepy.editor as mp
import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt

usage = 'Usage: python {} INPUT_DIR [--dir <directory>] [--help]'.format(__file__)
parser = argparse.ArgumentParser(description='This script is to generate images from a video.',
                                 usage=usage)
parser.add_argument('input_dir', action='store', nargs=None, 
                    type=str, help='Input directory.')
parser.add_argument('-c', '--config', action='store', default='config_gd.yaml',
                    help='Config file.')
parser.add_argument('-d', '--dir', action='store', nargs='?',
                    default='data', type=str, help='Directory of Output files.')
parser.add_argument('-f', '--flip', action='store_true',
                    default=False, help='Flip image.')
parser.add_argument('--noaudio', action='store_true',
                    default=False, help='Not generate audio data.')
args = parser.parse_args()
config = yaml.load(open('config_gd.yaml'), Loader=yaml.FullLoader)
videos = os.listdir(args.input_dir)
img_seq_file = os.path.join(args.dir, "img_sequence.txt")
img_seqf = open(img_seq_file, 'w')

for f in videos:
    path = os.path.join(args.input_dir, f)
    print("Load file:", path)
    videoclip = mp.VideoFileClip(path).subclip(config['time']['start'],
                                               config['time']['end'])
    out_dir = os.path.join(args.dir, os.path.splitext(os.path.basename(f))[0])
    os.makedirs(out_dir)
    ifiles = []
    img_conf = config['image']
    print('Start saving images...')
    for t, image in videoclip.iter_frames(fps=config['time']['fps'], with_times=True):
        t_indx = int(t * 1.0e3)
        sys.stdout.write('\rSave image {}'.format(t_indx))
        sys.stdout.flush()
        ifiles.append(os.path.join(out_dir, "%s%09d.jpg" % (img_conf['file_prefix'], t_indx)))
        image = image[:, :, ::-1]
        if args.flip:
            image = cv2.flip(image, 1)
        image = cv2.resize(image, (img_conf['width'], img_conf['height']))
        cv2.imwrite(ifiles[-1], image)

    # 学習用画像リストファイルを最後に作成して保存
    ilist_file = os.path.join(out_dir, "data_img_list.txt")
    print('\nSave %s' % ilist_file)
    with open(ilist_file, 'w') as f:
        f.write('\n'.join(ifiles))
    print("Done.")
    img_seqf.write(ilist_file + '\n')
    img_seqf.flush()

    # Save image data
    audioclip = videoclip.audio
    if args.noaudio or audioclip is None:
        continue
    ad_conf = config['audio']
    tms = []
    wvs = []
    for t, wv in audioclip.iter_frames(fps=ad_conf['sampling_rate'], with_times=True):
        tms.append(t)
        wvs.append(wv)

    # Calculate melspectrogram
    tms = np.array(tms)
    wvs = np.array(wvs)
    # force stereo to monoral
    wvs = np.expand_dims(librosa.to_mono(wvs.T).T, -1)
    n_window = int(ad_conf['window_size'] * ad_conf['sampling_rate'])
    hop_length = 1.0 / ad_conf['n_hop_length']
    n_hop_len = int(hop_length * ad_conf['sampling_rate'])
    n_ch = int(ad_conf['n_hop_length'] // config['time']['fps'])
    melsps = []
    for i in range(wvs.shape[1]):
        stft = np.abs(librosa.stft(wvs[:, i], n_fft=n_window, hop_length=n_hop_len))**2
        log_stft = librosa.power_to_db(stft)
        melsps.append(librosa.feature.melspectrogram(S=log_stft, n_mels=ad_conf['n_mels']))
    melsps = np.array(melsps).transpose((2, 1, 0))[:-1]
    melsps = np.split(melsps, melsps.shape[0] // n_ch)

    # Save audio data
    afiles = []
    tms = tms[::(n_hop_len * n_ch)]
    print('\nStart audio images...')
    for i, mls in enumerate(melsps):
        t_indx = int(tms[i] * 1.0e3)
        plt.figure()
        sys.stdout.write('\rSave audio {}'.format(t_indx))
        sys.stdout.flush()
        afiles.append(os.path.join(out_dir, "%s%09d.npy" % (ad_conf['file_prefix'], t_indx)))
        np.save(afiles[-1], mls.transpose((2, 1, 0)))
        librosa.display.specshow(mls[:, :, 0].T, sr=ad_conf['sampling_rate'], hop_length=n_hop_len, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.savefig(os.path.join(out_dir, "%s%09d.png" % (ad_conf['file_prefix'], t_indx)))
        plt.close()

    alist_file = os.path.join(out_dir, "data_audio_list.txt")
    print('\nSave %s' % alist_file)
    with open(alist_file, 'w') as f:
        f.write('\n'.join(afiles))
    print("Done.")
