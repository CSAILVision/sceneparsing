% This script demos how to use the pre-trained models to
% obtain the predicted segmentations

close all; clc; clear;
addpath(genpath('visualizationCode'));


% path to caffe (compile matcaffe first, or you could use python wrapper instead)
addpath 'yourcaffe/matlab' 

% select the pre-trained model. Use 'FCN' for 
% the Fully Convolutional Network or 'Dilated' for DilatedNet
% You can download the FCN model at 
% http://sceneparsing.csail.mit.edu/model/FCN_iter_160000.caffemodel
% and the DilatedNet model at
% http://sceneparsing.csail.mit.edu/model/DilatedNet_iter_120000.caffemodel
model_type = 'FCN'; %Dilated'
if (strcmp(model_type, 'FCN'))
	model_definition = 'models/deploy_FCN.prototxt';
	model_weights = 'FCN_iter_160000.caffemodel';
elseif (strcmp(model_type, 'Dilated')) 
	model_definition = 'models/deploy_DilatedNet.prototxt';
	model_weights = 'DilatedNet_iter_120000.caffemodel';
end
disp(model_definition)
prediction_folder = sprintf('predictions_%s', model_type);

% initialize the network
net = caffe.Net(model_definition, model_weights, 'test');

% path to image(.jpg) and annotation(.png) and generated prediction(.png)
pathImg = fullfile('sampleData', 'images');
pathAnno = fullfile('sampleData', 'annotations');
pathPred = fullfile('sampleData', prediction_folder);

if (~exist(pathPred, 'dir'))
	mkdir(pathPred);
end

% load class names
load('objectName150.mat');
% load pre-defined colors 
load('color150.mat');

filesImg = dir(fullfile(pathImg, '*.jpg'));
for i = 1: numel(filesImg)
    % read image
    fileImg = fullfile(pathImg, filesImg(i).name);
    fileAnno = fullfile(pathAnno, strrep(filesImg(i).name, '.jpg', '.png'));
    filePred = fullfile(pathPred, strrep(filesImg(i).name, '.jpg', '.png'));

    im = imread(fileImg);
    imAnno = imread(fileAnno);
  	
    % resize image to fit model description
    im_inp = double(imresize(im, [384,384])); 

    % change RGB to BGR
    im_inp = im_inp(:,:,end:-1:1);

    % substract mean and transpose
    im_inp = cat(3, im_inp(:,:,1)-109.5388, im_inp(:,:,2)-118.6897, im_inp(:,:,3)-124.6901);
    im_inp = permute(im_inp, [2,1,3]);

    % obtain predicted image and resize to original size
    imPred = net.forward({im_inp});
    [~, imPred] = max(imPred{1},[],3);
    imPred = uint8(imPred')-1;
    imPred = imresize(imPred, [size(im,1), size(im,2)], 'nearest');
    imwrite(imPred, filePred);
 
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
