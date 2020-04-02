import os
import regex as re
import itertools

import cv2
import pytesseract
from pytesseract import Output
import pandas as pd
import numpy as np

from PIL import Image

def get_extracted_dict_single_key(search_dict, text_df,flatten_text):

  filtered_df = text_df.copy()
  filtered_df["text"] = filtered_df["text"].apply(lambda x:x.strip())
  filtered_df = filtered_df.loc[filtered_df["text"] != "\t"]
  filtered_df = filtered_df.loc[filtered_df["text"] != "\n"]
  filtered_df = filtered_df.loc[filtered_df["text"] != "\r"]
  filtered_df = filtered_df.loc[filtered_df["text"] != ""]
  filtered_df.index = np.arange(filtered_df.shape[0])

  if filtered_df.shape[0] != len(flatten_text.split()):
    print("dimension mismatch")

  key = list(search_dict.keys())[0]
  extracted_dict = {key:[]}
  pattern = re.compile(search_dict[key])

  for match in pattern.finditer(flatten_text):
    start = match.start()
    end = match.end()

    j = start
    for j in range(start,0,-1):
      if flatten_text[j] in [" ","\n","\r","\t"]:
        break
    
    i = end
    for i in range(end,len(flatten_text)):
      if flatten_text[i] in [" ","\n","\r","\t"]:
        break
    
    number_words_df = len(flatten_text[j:i].strip().split())
    first_word_index = len(flatten_text[:j].split())
    
    sub_df = filtered_df.iloc[first_word_index:first_word_index+number_words_df]

    left = min(sub_df["left"].values)
    top = min(sub_df["top"].values)
    height = max(sub_df["height"].values)
    last_row = sub_df.iloc[-1]
    width = last_row["left"]+last_row["width"]-left

    text_match = match.group()

    text_in_df = " ".join(sub_df["text"].values)
    #start and end indexes of text_match in text_in_df
    start_string = text_in_df.find(text_match)
    width_per_letter = width/len(text_in_df)

    left = left + start_string*width_per_letter
    width = width_per_letter*len(text_match)

    extracted_dict[key].append({"text":text_match,"left":left,"top":top,"width":width,"height":height})

  return extracted_dict


