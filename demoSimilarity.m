% This script demos how to use the similarity matrix 
close all; clc; clear;

% load the similarity data 
load('human_semantic_similarity.mat');

% similarity(i, j) indicates how good is predicting class j given the groundtruth class i. 
fprintf('Groundtruth class: %s, Predicted class: %s, Score: %.4f\n', object_names{129}, object_names{61}, similarity(129,61));

% predicting the correct class gives a score of 1.
fprintf('Groundtruth class: %s, Predicted class: %s, Score: %.4f\n', object_names{2}, object_names{2}, similarity(2,2));

% when the two classes are unrelated the score is 0.
fprintf('Groundtruth class: %s, Predicted class: %s, Score: %.4f\n', object_names{7}, object_names{9}, similarity(7,9));


