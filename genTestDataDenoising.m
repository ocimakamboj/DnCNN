sourceFolder = fullfile('TestDataSet','BSDS500_RR','Coloured','BSD68','*.jpg');
variableN = 'Sigma-50';
sigma = 50/255;
destFolder1 = fullfile('Results','Denoising','BSD68',variableN,'Original');
destFolder2 = fullfile('Results','Denoising','BSD68',variableN,'TestData');
nameSuffix1 = 'original_';
nameSuffix2 = 'test_';

if ~exist(destFolder1,'dir')
    mkdir(destFolder1);
end

if ~exist(destFolder2,'dir')
    mkdir(destFolder2);
end

filepaths = [];
filepaths = cat(1,filepaths,dir(fullfile(sourceFolder)));
sizeA = zeros(length(filepaths),2);


for i = 1:length(filepaths)
    source = fullfile(filepaths(i).folder,filepaths(i).name);
    A = imread(source);
    [height,width,c] = size(A);
    sizeA(i,1) = height;
    sizeA(i,2) = width;
end

bins=unique(sizeA,'rows');
binsCount = zeros(size(bins,1),1);
testDataCollective = cell(size(bins,1),1);
storedIndex = zeros(length(filepaths),1);

for i = 1:length(filepaths)
    for j = 1:size(bins,1)
        if(sizeA(i,:) == bins(j,:))
            binsCount(j) = binsCount(j) + 1;
            storedIndex(i) = j;
            break;
        end
    end
end

binsCountCum = cumsum(binsCount);

for j = 1:size(bins,1)
    testDataCollective{j,1} = zeros(binsCount(j),bins(j,1),bins(j,2),3,'single');
end

counts = ones(size(bins,1),1);
for i = 1:length(filepaths)
    source = fullfile(filepaths(i).folder,filepaths(i).name);
    A = imread(source);
    [height,width,c] = size(A);
    I = A;
    
    index = counts(storedIndex(i,1)) + binsCountCum(storedIndex(i,1)) - binsCount(storedIndex(i,1));
    newName1 = [nameSuffix1,num2str(index,'%.2d\n'),'.jpg'];
    destination1 = fullfile(destFolder1,newName1);
    imwrite(I,destination1);
    
    I_N = imnoise(I,'gaussian',0,sigma*sigma);
    newName2 = [nameSuffix2,num2str(index,'%.2d\n'),'.jpg'];
    destination2 = fullfile(destFolder2,newName2);
    imwrite(I_N,destination2);

    testData = im2single(I_N);
    
    testDataCollective{storedIndex(i,1)}(counts(storedIndex(i,1)),:,:,:) = testData;
    counts(storedIndex(i,1)) = counts(storedIndex(i,1)) + 1;
    
%     name = ['testData',num2str(i,'%.2d\n'),'.mat'];
%     save(fullfile(destFolder3,name),'testData');
end

for j =1:size(bins,1)
    testData = testDataCollective{j,1};
    name = ['testDataCollective',num2str(j,'%.2d\n'),'.mat'];
    save(fullfile(destFolder2,name),'testData');
end

