#!/usr/bin/env bash

wget -O caltech101.tar.gz http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
tar -xf caltech101.tar.gz
mv 101_ObjectCategories caltech101
