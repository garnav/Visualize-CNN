# scoring.py
# Arnav Ghosh
# 27th March 2020

import copy
import cv2
import numpy as np
import plotly.graph_objects as go
from scipy import ndimage

from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
import spacy

# TODO: Need to wrap functions into class
#nlp = spacy.load("en_core_web_lg")

def get_features(detector, main_image, sec_image):
    main_image = cv2.GaussianBlur(main_image,(3,3),0)
    sec_image = cv2.GaussianBlur(sec_image,(3,3),0)

    if detector.lower().strip() == "sift":
        feat_detector = cv2.xfeatures2d.SIFT_create()
    elif detector.lower().strip() == "surf":
        feat_detector = cv2.xfeatures2d.SURF_create()
    else:
        raise Exception("Unknown detector chosen.")

    kp1, des1 = feat_detector.detectAndCompute(main_image, None)
    kp2, des2 = feat_detector.detectAndCompute(sec_image, None)

    return kp1, kp2, des1, des2

def score_unsym_matches(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    acc_matches = list(filter(lambda x : x[0].distance < 0.7 * x[1].distance, matches))
    return len(acc_matches) / len (matches)

def score_sym_matches(des1, des2):
    des1_des2_pct = score_unsym_matches(des1, des2)
    des2_des1_pct = score_unsym_matches(des2, des1)

    return (2 * des1_des2_pct * des2_des1_pct) / (des1_des2_pct + des2_des1_pct)

def score_cross_matches(des1, des2):
    bf = cv2.BFMatcher(crossCheck=True)
    matches = bf.knnMatch(des1, des2, k=1)

    return len(matches) / min(len(des1), len(des2))

def label_semantic_score(label1, label2, method):
    if method.lower().strip() == "embed":
        score = nlp(label1).similarity(nlp(label2))
    elif method.lower().strip() == "jc":
        synset1 = wn.synset(f"{label1}.n.01")
        synset2 = wn.synset(f"{label2}.n.01")
        brown_ic = wordnet_ic.ic('ic-brown.dat')
        score = synset1.jcn_similarity(synset2, brown_ic)
    else:
        raise Exception("Unknown method chosen.")

    return score

def label_visual_score(label1, label2):
    pass