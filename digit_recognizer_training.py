#!/usr/bin/python
# -*- coding: utf8 -*-

#################################################################
# File name: digit_recognizer_training.py                       #
# Description: Train digits on an image and save them as files. #
# Version: 0.0.1                                                #
# Author: Gökhan Sari                                           #
# E-mail: g-sari@g-sari.com                                     #
#################################################################

import sys
import cv2
import numpy as np
from picture import Pic


class DigitRecognizerTraining:
    """Class used to train digits on an image"""

    def __init__(self):
        self.training_pics = [Pic(), Pic(pic_name="ocr_insurance_card_train_2.jpg", contour_dimension_from_h=21, contour_dimension_to_h=28)]

    def train(self):
        """Method to train digits"""
        # Loop all images to train
        for training_pic in self.training_pics:
            im = cv2.imread(training_pic.pic_name)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 1)
            _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            samples = np.empty((0, 100))
            responses = []
            keys = [i for i in range(48, 58)]

            for cnt in contours:
                if cv2.contourArea(cnt) > (training_pic.contour_dimension_to_h * 2):
                    [x, y, w, h] = cv2.boundingRect(cnt)
                    # print("contour:" w, h)
                    if h > training_pic.contour_dimension_from_h and h < training_pic.contour_dimension_to_h:
                        cv2.rectangle(im, (x - 1, y - 1), (x + 1 + w, y + 1 + h), (0, 0, 255), 1)
                        roi = thresh[y:y + h, x:x + w]
                        roismall = cv2.resize(roi, (10, 10))
                        cv2.imshow('Training: Enter digits displayed in the red rectangle!', im)
                        key = cv2.waitKey(0)

                        if key == 27:  # (escape to quit)
                            self.save_data(samples, responses)
                            cv2.destroyAllWindows()
                            sys.exit()
                        elif key in keys:  # (append data)
                            responses.append(int(chr(key)))
                            sample = roismall.reshape((1, 100))
                            samples = np.append(samples, sample, 0)
        # Save collected data
        self.save_data(samples, responses)

    @staticmethod
    def save_data(samples, responses):
        """Method to save trained data"""
        responses = np.array(responses, np.float32)
        responses = responses.reshape((responses.size, 1))
        np.savetxt('ocr_training.data', samples)
        np.savetxt('ocr_responses.data', responses)
        print "training complete"

# Start the training process
if __name__ == '__main__':
    DigitRecognizerTraining().train()
