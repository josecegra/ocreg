import os
import re
import itertools
import json

import random

import cv2
import pytesseract
from pytesseract import Output
import pandas as pd
import numpy as np

from PIL import Image

from src.document_class import DocumentOCR

#before dumping
#doc.flatten_text = flatten_text.encode("utf-8")
#after dumping
#flatten_text = flatten_text.encode("utf-8")
#after loading
#flatten_text = flatten_text.encode("utf-8")


def main():

  print("started main")
  exec_path = os.getcwd()

  file_list = os.listdir(exec_path)
  for filename in file_list:
    if ".png" in filename or ".pdf" in filename or ".jpg" in filename:
      path_img = os.path.join(exec_path,filename)
      single_file(path_img)

  return 

"""
  search_dict = {
            "numbers":"\d+",
            "jose":"[Jj]ose|JOSE",
            "dates":"\d{2}[\-\\\/]\d{2}[\-\\\/]\d{2,4}",
            "ml2":"ine learn",
            "email":"[a-zA-Z]+@[a-z]+\.?[a-z]+",
            "dni":"\d{8,9}[A-Z]",
            "cea": "[0-9A-Z]{5}\-[0-9A-Z]{5}\-[0-9A-Z]{5}\-[0-9A-Z]{5}\-[0-9A-Z]{5}\-[0-9A-Z]{5}"
            }

"""

def single_file(path_img):

  def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,img = cv2.threshold(img,180,255,cv2.THRESH_BINARY)
    kernel = np.ones((2,2),np.uint8)
    img = cv2.erode(img,kernel,iterations = 1)

    return img

  doc = DocumentOCR(path_img)
  doc.remove_tmp()
  doc = doc.apply_preprocessing(preprocessing)
  doc = doc.apply_ocr()
  df = doc.df

  search_dict = {
            "dates":r'\d{1,2}\/\d{1,2}\/\d{2,4}',
            "total":r"\s{2}Total",
            "email":r"[a-zA-Z]+@[a-z]+\.?[a-z]+",
            "numbers":r"\d+",

            "prices": r"\d+[,\.]\d+"
            }

  extracted_dict = df.get_extracted_dict(search_dict)
  print(doc.flatten_text)
  print(extracted_dict)
  highlighted_img= doc.highlight_image(extracted_dict,save=True)

  return






    


    






       

 


    

    
            

