function plotResult(im, imPred, imAnno, objectNames, colors, filename)
%% This function plots single image results
if nargin<5
    % use random color instead
    colors = im2uint8(hsv(255));
    rng(1);
    colors = colors(randperm(255), :);
end
if nargin<6
    filename = '';
end

% color encoding
rgbPred = colorEncode(imPred, colors);
rgbLab = colorEncode(imAnno, colors);

% colormaps
colormap = colorMap(imPred, imAnno, objectNames);
    
% plot
set(gcf, 'Name', [filename ' [Press any key to the next image...]'],  'NumberTitle','off');

subplot(221);
imshow(im); title('Image');
subplot(222);
imshow(rgbPred); title('Prediction');
subplot(223);
imshow(colormap); title('Colormap');
subplot(224);
imshow(rgbLab); title('Annotation (GT)');
drawnow;