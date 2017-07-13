#!/usr/bin/python
# -*- coding: utf8 -*-

#################################################################
# File name: digit_recognizer_training.py                       #
# Description: Recognize and test digits on an image.           #
# Version: 0.0.1                                                #
# Author: Gökhan Sari                                           #
# E-mail: g-sari@g-sari.com                                     #
#################################################################

import cv2
import numpy as np
from picture import Pic


class DigitRecognizerTesting:
    """Class used to test digits on an image"""

    def __init__(self):
        self.image_to_test = Pic(pic_name="ocr_insurance_card_test_2.jpg", contour_dimension_from_h=21, contour_dimension_to_h=28)
        self.load_training_data()
        self.model = cv2.ml.KNearest_create()
        self.model.train(self.samples, cv2.ml.ROW_SAMPLE, self.responses)

    def load_training_data(self):
        self.samples = np.loadtxt('ocr_training.data', np.float32)
        self.responses = np.loadtxt('ocr_responses.data', np.float32)
        self.responses = self.responses.reshape((self.responses.size, 1))

    def test(self):
        im = cv2.imread(self.image_to_test.pic_name)
        out = np.zeros(im.shape, np.uint8)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)
        insurance_card_number_output = "";
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) > self.image_to_test.contour_dimension_to_h:
                [x, y, w, h] = cv2.boundingRect(cnt)
                if h > self.image_to_test.contour_dimension_from_h and h < self.image_to_test.contour_dimension_to_h:
                    cv2.rectangle(im, (x - 1, y - 1), (x + 1 + w, y + 1 + h), (0, 255, 0), 1)
                    roi = thresh[y:y + h, x:x + w]
                    roismall = cv2.resize(roi, (10, 10))
                    roismall = roismall.reshape((1, 100))
                    roismall = np.float32(roismall)
                    retval, results, neigh_resp, dists = self.model.findNearest(roismall, k=1)
                    string = str(int((results[0][0])))
                    insurance_card_number_output += string
                    cv2.putText(out, string, (x, y + h), 0, 1, (0, 255, 0))

        reversed_insurance_card_number_output = insurance_card_number_output[::-1]
        print("Detected insurance card number: " + reversed_insurance_card_number_output)
        cv2.imshow('input', im)
        cv2.imshow('output', out)
        cv2.waitKey(0)

# Start the training process
if __name__ == '__main__':
    DigitRecognizerTesting().test()
