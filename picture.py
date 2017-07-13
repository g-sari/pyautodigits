#!/usr/bin/python
# -*- coding: utf8 -*-

#################################################################
# File name: picture.py                                         #
# Description: Represents an image to train or test.            #
# Version: 0.0.1                                                #
# Author: Gökhan Sari                                           #
# E-mail: g-sari@g-sari.com                                     #
#################################################################

class Pic:
    """Class that represents a picture to train or test"""

    def __init__(self, pic_name="ocr_insurance_card_train_1.png", contour_dimension_from_h=14, contour_dimension_to_h=23):
        self.pic_name = pic_name
        self.contour_dimension_from_h = contour_dimension_from_h
        self.contour_dimension_to_h = contour_dimension_to_h