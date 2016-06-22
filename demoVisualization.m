% This script demos how to use colorEncode() to visualize label maps
close all; clc; clear;

% Paths and options
addpath(genpath('visualizationCode'));

% path to image(.jpg), prediction(.png) and annotation(.png)
pathImg = fullfile('sampleData', 'images');
pathPred = fullfile('sampleData', 'predictions');
pathLab = fullfile('sampleData', 'annotations');

% load class names
load('objectName150.mat');
% load pre-defined colors 
load('color150.mat');

filesPred = dir(fullfile(pathPred, '*.png'));
for i = 1: numel(filesPred)
    % read image
    fileImg = fullfile(pathImg, strrep(filesPred(i).name, '.png', '.jpg'));
    filePred = fullfile(pathPred, filesPred(i).name);
    fileLab = fullfile(pathLab, filesPred(i).name);
    im = imread(fileImg);
    imPred = imread(filePred);
    imAnno = imread(fileLab);
    
    % color encoding
    rgbPred = colorEncode(imPred, colors);
    rgbAnno = colorEncode(imAnno, colors);
    
    % colormaps
    colormap = colorMap(imPred, imAnno, objectNames);
    
    % plot
    set(gcf, 'Name', [fileImg ' [Press any key to the next image...]'],  'NumberTitle','off');
    
    subplot(231);
    imshow(im); title('Image');
    subplot(232);
    imshow(imPred); title('Prediction-gray');
    subplot(233);
    imshow(imAnno); title('Annotation-gray');
    subplot(234);
    imshow(colormap); title('Colormap');
    subplot(235);
    imshow(rgbPred); title('Prediction-color');
    subplot(236);
    imshow(rgbAnno); title('Annotation-color');

    waitforbuttonpress;
    
end




