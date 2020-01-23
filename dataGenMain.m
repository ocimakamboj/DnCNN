PAR = [];
f=0.32; %f=0.32 for creating validation dataset
noImages = 400*f/4; 
noImagesPerBatch = 8000*f/4;
PAR.sizeInputData = 60;
PAR.colorChannel = 3;
PAR.stride = 10;
PAR.batchSize = 128;
PAR.numberPatches = PAR.batchSize*noImagesPerBatch;
noPatches = noImages*floor((321-PAR.sizeInputData)/PAR.stride + 1)*floor((481-PAR.sizeInputData)/PAR.stride + 1);
PAR.N_AUG = ceil(PAR.numberPatches/noPatches);
PAR.sourceFolder = fullfile('BSDS500_RR','Coloured','train','*.jpg');
if f~=1
    PAR.sourceFolder = fullfile('BSDS500_RR','Coloured','val','*.jpg');
end
PAR.Space = noPatches*PAR.N_AUG;

%%
[inputData,labels] = genPatches(PAR);

if(f~=1)
    inputDataVal = inputData;
    clear inputData;
    labelsVal = labels;
    clear labels;
end

%%
path = fullfile('Data');
if ~exist(path,'dir')
    mkdir(path);
end

if f==1
    save(fullfile(path,'inputData.mat'),'inputData');
    save(fullfile(path,'labels.mat'),'labels');
else
    save(fullfile(path,'inputDataVal.mat'),'inputDataVal');
    save(fullfile(path,'labelsVal.mat'),'labelsVal');
end
    