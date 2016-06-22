function colormap = colorMap(imPred, imAnno, objectnames)
%% This function encodes label maps into rgb images for better visualization
% imPred, imAnno: [h, w]
% objectnames: {n, 1} where n is the number of classes
% colormap: [h, w, 3]

colormap = cell(2,8);

idxUnique = unique([imPred, imAnno])';
cnt = 0;
for idx = idxUnique
    if idx==0   
        continue;   
    else
        cnt = cnt + 1;
        colormap{cnt} = imread([objectnames{idx} '.jpg']);
    end
    
    if cnt>= numel(colormap)
        break; 
    end
end

for i = cnt+1 : numel(colormap)
    colormap{i} = 255 * ones(30, 150, 3, 'uint8');
end

colormap = cell2mat(colormap');