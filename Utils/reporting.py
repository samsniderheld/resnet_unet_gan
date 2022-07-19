import cv2
import numpy as np
import os
import csv
import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt


def generate_images(model, test_input, path, epoch):

  prediction = model(test_input)
    
  display_list = [test_input[0], prediction[0]]

  output_image = np.uint8(prediction[0]) * 255

  output_path = os.path.join(path,f"epoch_{epoch:04d}_image.jpg")

  cv2.imwrite(output_path,output_image)


def save_experiment_history(args, history, path):

  experiment = {

      'notes': args.notes,
      'number_of_epochs': args.num_epochs,
      'batch_size': args.batch_size,
      'loss_history': history

  }

  file_name = args.experiment_name + ".json"

  output_path = os.path.join(path,file_name)

  with open(output_path, 'w') as outfile:
    json.dump(experiment, outfile)









