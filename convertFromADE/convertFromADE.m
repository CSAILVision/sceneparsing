% This function converts an annotation image from ADE dataset format to
% scene parsing challenge format

% input args: filename_label_input, filename_label_output
% example: convertFromADE('ADE_train_00000970_raw.png','ADE_train_00000970_challenge.png')

function convertFromADE(fileLabIn, fileLabOut)
    % load in mapping
    indexMapping = dlmread('mapFromADE.txt');
    
    % read in label map
    lab = imread(fileLabIn);
    
    % convert
    labOut = convert(lab, indexMapping);

    % save to file
    imwrite(labOut, fileLabOut);
end

function labOut = convert(lab, indexMapping)

    % resize
    [h, w, ~] = size(lab);
    h_new = h;
    w_new = w;
    if h<w && h>512
        h_new = 512;
        w_new = round(w/h*512);
    elseif w<h && w>512
        h_new = round(h/w*512);
        w_new = 512;
    end
    lab = imresize(lab, [h_new, w_new], 'nearest');
    
    % map index
    labADE = (uint16(lab(:,:,1))/10)*256+uint16(lab(:,:,2));
    labOut = zeros(size(labADE), 'uint8');
    
    classes_unique = unique(labADE)';
    for cls = classes_unique
        if sum(cls==indexMapping(:,2))>0
            labOut(labADE==cls) = indexMapping(cls==indexMapping(:,2), 1);
        end
    end

end
