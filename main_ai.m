
srcDir='S:\Engineering\AI Datasets\PresetFreeImaging\Test3\';
dstDir='S:\Engineering\AI Datasets\PresetFreeImaging\Test3\';

imageFolder = fullfile(dstDir(1:end-1));

imds = imageDatastore(imageFolder, 'LabelSource', 'foldernames', 'IncludeSubfolders', true);

classNames = categories(imds.Labels)
numClasses = numel(classNames)

tbl = countEachLabel(imds);

[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomize');

%%
net = imagePretrainedNetwork("googlenet", NumClasses=numClasses); 

learnables = net.Learnables;
trainNode = learnables{end,'Layer'};

net = setLearnRateFactor(net, trainNode, Weights=10);
net = setLearnRateFactor(net, trainNode, Bias=10);

inputSize = net.Layers(1).InputSize;

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);

%%
miniBatchSize = 10;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');

%%
tic
net = trainnet(augimdsTrain, net, "crossentropy", options);
toc
%%
tic
prob = minibatchpredict(net,augimdsValidation);
toc
%%
validationPredict = scores2label(prob,classNames);
validationTruth = imdsValidation.Labels;

kp=1;
figure(kp); kp=kp+1;
plotconfusion(validationTruth, validationPredict); % output is prediction, target is ground truth  % confusionchart(validationTruth,validationPredict);

[confMat, gorder] = confusionmat(validationTruth, validationPredict); % [true x predicted]

probObjectIsCorrectlyIdentified_ratio = confMat ./ sum(confMat,2); % if object is x, what is the percentage time it will be predicted to be x

probPredictionIsCorrect_ratio = confMat ./ sum(confMat,1); % if object is predicted to be x, what is the percentage time the object is really x

overall_accuracy = sum(diag(confMat))/sum(confMat(:));

%%
filename = "presetFreeImaging_googlenet_1.onnx";
exportONNXNetwork(net,filename,'BatchSize',1);

