import os
import re
import itertools
import json
import random
import shutil
import cv2
import pytesseract
from pytesseract import Output
import pandas as pd
import numpy as np
from PIL import Image

from pdf2image import convert_from_path

from src.tools import get_extracted_dict_single_key

class DocumentOCR(object):

  def __init__(self,path_img):

    self.path_img = path_img
    if os.path.exists(self.path_img):
      self.path_data = os.path.split(self.path_img)[0]
      self.filename = os.path.split(self.path_img)[1]

      if self.filename[-4:] == ".png" or self.filename[-4:] == ".jpg":
        self.raw_img = cv2.imread(self.path_img)
        self.img = self.raw_img
      elif self.filename[-4:] == ".pdf":
        self.raw_img = np.array(convert_from_path(self.path_img)[0])   
        self.img = self.raw_img
   

      self.path_tmp = os.path.join(self.path_data,"tmp")
      self.tmp_bool = False
      if os.path.exists(self.path_tmp):
        bool_file_1 = "{}_df.json".format(self.filename) in os.listdir(self.path_tmp)
        bool_file_2 = "{}_text_dict.json".format(self.filename) in os.listdir(self.path_tmp)
        bool_file_3 = "{}_ft.json".format(self.filename) in os.listdir(self.path_tmp)
        if bool_file_1 and bool_file_2 and bool_file_3:
          self.tmp_bool = True
      else:
        os.chdir(self.path_data)
        os.mkdir(self.path_tmp)
    else:
      print("no image found")

  def remove_tmp(self):
    if self.tmp_bool:
      os.chdir(self.path_tmp)
      for filename in os.listdir(self.path_tmp):
        os.remove(filename)
      self.tmp_bool = False
      

  def apply_preprocessing(self,func):
    img = np.array(self.img)
    img = func(img)
    return self


  def apply_ocr(self):

    self.df = pd.DataFrame()
    self.flatten_text = ""
    if self.tmp_bool:
      if "{}_df.json".format(self.filename) in os.listdir(self.path_tmp):
        tmp_df_path = os.path.join(self.path_tmp,"{}_df.json".format(self.filename))
        with open(tmp_df_path,"r") as f:
          text_df = pd.DataFrame(json.load(f))
          text_df.index = [int(i) for i in text_df.index]
          text_df = text_df.sort_index()
          
      if "{}_text_dict.json".format(self.filename) in os.listdir(self.path_tmp):
        tmp_text_dict_path = os.path.join(self.path_tmp,"{}_text_dict.json".format(self.filename))
        with open(tmp_text_dict_path,"r",encoding = 'utf-8') as f:
          text_dict = json.load(f)
          text_df["text"] = text_dict["text"]
          self.df = DataFrameOCR(text_df)

      if "{}_ft.json".format(self.filename) in os.listdir(self.path_tmp):
        tmp_ft_path = os.path.join(self.path_tmp,"{}_ft.json".format(self.filename))
        with open(tmp_ft_path,"r",encoding='utf-8') as f:
          flatten_text = json.load(f)["flatten_text"]
          self.flatten_text = flatten_text

    else:
      #getting document dataframe
      self.df = DataFrameOCR(pytesseract.image_to_data(self.img,nice=1,output_type=Output.DATAFRAME))
      #drop tabs
      self.df = self.df.dropna()
      self.df.index = np.arange(self.df.shape[0])
      #normalize df
      x_max = self.img.shape[0]
      y_max = self.img.shape[1]
      #x_max,y_max = self.img.size
      self.df["left"] = self.df["left"]/x_max
      self.df["width"] = self.df["width"]/x_max
      self.df["top"] = self.df["top"]/y_max
      self.df["height"] = self.df["height"]/y_max


      #getting flatten_text
      self.flatten_text = pytesseract.image_to_string(self.img)

      #print("saving tmp")
      os.chdir(self.path_tmp)
      tmp_text_dict_path = os.path.join(self.path_tmp,"{}_text_dict.json".format(self.filename))
      with open(tmp_text_dict_path,"w",encoding='utf-8') as f:
        json.dump({"text":list(self.df["text"].values)},f,ensure_ascii=False)
      
      save_df = self.df.copy(deep=True)
      save_df.pop("text")
      tmp_df_path = os.path.join(self.path_tmp,"{}_df.json".format(self.filename))
      with open(tmp_df_path,"w") as f:
        json.dump(save_df.to_dict(),f)

      tmp_ft_path = os.path.join(self.path_tmp,"{}_ft.json".format(self.filename))
      with open(tmp_ft_path,"w",encoding='utf-8') as f:
        json.dump({"flatten_text":self.flatten_text},f,ensure_ascii=False)

    return self

  def highlight_image(self, extracted_dict,save=False):

      x_max = self.raw_img.shape[0]
      y_max = self.raw_img.shape[1]

      
      img = self.raw_img
      highlighted_img = img

      for key in extracted_dict.keys(): 
          color = (0,255,0)
          if extracted_dict[key]:
            for _dict in extracted_dict[key]:

                _dict["left"] = max(0,_dict["left"]-0.01)
                _dict["top"] = max(0,_dict["top"]-0.01)
                _dict["width"] += 0.01
                _dict["height"] += 0.01

                left = _dict["left"]*x_max
                top = _dict["top"]*y_max
                width = _dict["width"]*x_max
                height = _dict["height"]*y_max

                # Initialize black image of same dimensions for drawing the rectangles
                blk = np.zeros(img.shape, np.uint8)
                # Draw rectangles
                cv2.rectangle(blk, (int(left), int(top)), (int(left+width), int(top+height)), color, cv2.FILLED)
                # Generate result by blending both images (opacity of rectangle image is 0.25 = 25 %)
                highlighted_img = cv2.addWeighted(highlighted_img, 1.0, blk, 0.75, 1)
                

      if save:    
        path_highlighted_img = os.path.join(self.path_tmp,"{}_highlighted.png".format(self.filename))
        cv2.imwrite(path_highlighted_img,highlighted_img)
      output_img = Image.fromarray(highlighted_img.astype(np.uint8))

      return output_img



class DataFrameOCR(pd.DataFrame):

    @property
    def _constructor(self):
        return DataFrameOCR

    def get_extracted_dict(self,search_dict):
      flatten_text = " ".join(self["text"].values)
      extracted_dict = {}
      for k,v in search_dict.items():
        single_key_search_dict = {k:v}
        single_key_extracted_dict = get_extracted_dict_single_key(single_key_search_dict, self, flatten_text)
        extracted_dict.update(single_key_extracted_dict)
      return extracted_dict


