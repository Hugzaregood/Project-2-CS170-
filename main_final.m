clc;
clear;

% ----------------------------
% ------- LOAD DATASET -------
% ----------------------------
filename = input('Type in the name of the file to test: ', 's');
data = load(filename);

[numInstances, numCols] = size(data);
numFeatures = numCols - 1;

fprintf('This dataset has %d features (not including the class attribute), with %d instances.\n\n', ...
    numFeatures, numInstances);

% ----------------------------
% ---- CHOOSE ALGORITHM ------
% ----------------------------
fprintf('Type the number of the algorithm you want to run.\n');
fprintf('1) Forward Selection\n');
fprintf('2) Backward Elimination\n');
algorithmChoice = input('Enter choice: ');

fprintf('\nBeginning search.\n\n');

% ----------------------------
% -- INITIALIZE FOR RESULTS --
% ----------------------------
levels = [];
featureSets = {};
accuracies = [];
bestOverallSet = [];
bestOverallAccuracy = 0;

% ==========================================================
% =============== FORWARD SELECTION =========================
% ==========================================================
if algorithmChoice == 1

    currentSet = [];   % start with no features

    for level = 1:numFeatures

        featureToAddAtThisLevel = -1;
        bestAccuracySoFar = 0;

        for f = 1:numFeatures

            if ismember(f, currentSet)
                continue;
            end

            candidateSet = [currentSet f];
            accuracy = evaluateSubset(data, candidateSet);

            fprintf('Using feature(s) %s accuracy is %.4f\n', ...
                setToString(candidateSet), accuracy);

            if accuracy > bestAccuracySoFar
                bestAccuracySoFar = accuracy;
                featureToAddAtThisLevel = f;
            end
        end

        currentSet = [currentSet featureToAddAtThisLevel];

        fprintf('\nFeature set %s was best, accuracy is %.4f\n\n', ...
            setToString(currentSet), bestAccuracySoFar);

        % Save results for report
        levels(end+1) = level;
        featureSets{end+1} = setToString(currentSet);
        accuracies(end+1) = bestAccuracySoFar;

        % Track best overall subset
        if bestAccuracySoFar > bestOverallAccuracy
            bestOverallAccuracy = bestAccuracySoFar;
            bestOverallSet = currentSet;
        else
            fprintf('(Warning, Accuracy has decreased! Continuing search in case of local maxima)\n\n');
        end
    end

    csvName = 'forward_selection_results.csv';
    plotTitle = 'Forward Selection Accuracy';

% ==========================================================
% =============== BACKWARD ELIMINATION ======================
% ==========================================================
elseif algorithmChoice == 2

    currentSet = 1:numFeatures;   % start with all features

    % Evaluate all features first
    initialAccuracy = evaluateSubset(data, currentSet);
    bestOverallSet = currentSet;
    bestOverallAccuracy = initialAccuracy;

    fprintf('Using feature(s) %s accuracy is %.4f\n\n', ...
        setToString(currentSet), initialAccuracy);

    levels(end+1) = length(currentSet);
    featureSets{end+1} = setToString(currentSet);
    accuracies(end+1) = initialAccuracy;

    for level = numFeatures:-1:2

        featureToRemoveAtThisLevel = -1;
        bestAccuracySoFar = 0;
        bestSetThisLevel = [];

        for idx = 1:length(currentSet)

            candidateSet = currentSet;
            candidateSet(idx) = [];   % remove one feature

            accuracy = evaluateSubset(data, candidateSet);

            fprintf('Using feature(s) %s accuracy is %.4f\n', ...
                setToString(candidateSet), accuracy);

            if accuracy > bestAccuracySoFar
                bestAccuracySoFar = accuracy;
                featureToRemoveAtThisLevel = idx;
                bestSetThisLevel = candidateSet;
            end
        end

        currentSet = bestSetThisLevel;

        fprintf('\nFeature set %s was best, accuracy is %.4f\n\n', ...
            setToString(currentSet), bestAccuracySoFar);

        % Save results for report
        levels(end+1) = length(currentSet);
        featureSets{end+1} = setToString(currentSet);
        accuracies(end+1) = bestAccuracySoFar;

        % Track best overall subset
        if bestAccuracySoFar > bestOverallAccuracy
            bestOverallAccuracy = bestAccuracySoFar;
            bestOverallSet = currentSet;
        else
            fprintf('(Warning, Accuracy has decreased! Continuing search in case of local maxima)\n\n');
        end
    end

    csvName = 'backward_elimination_results.csv';
    plotTitle = 'Backward Elimination Accuracy';

% ==========================================================
% ================= INVALID OPTION ==========================
% ==========================================================
else
    fprintf('Invalid choice. Please run the program again and enter 1 or 2.\n');
    return;
end

% ----------------------------
% ------- FINAL RESULT -------
% ----------------------------
fprintf('Finished search!! The best feature subset is %s, which has an accuracy of %.4f\n', ...
    setToString(bestOverallSet), bestOverallAccuracy);

% ----------------------------
% ---- CREATE RESULTS TABLE ---
% ----------------------------
resultsTable = table(levels', featureSets', accuracies', ...
    'VariableNames', {'Level', 'FeatureSet', 'Accuracy'});

disp(resultsTable)

% ----------------------------
% ------ SAVE RESULTS CSV ----
% ----------------------------
writetable(resultsTable, csvName);

% ----------------------------
% --------- PLOT -------------
% ----------------------------
figure
plot(levels, accuracies, '-o')
xlabel('Feature Set Size')
ylabel('Accuracy')
title(plotTitle)
grid on

% ==========================================================
% ================= HELPER FUNCTIONS ========================
% ==========================================================
function accuracy = evaluateSubset(data, candidateSet)
    numInstances = size(data, 1);
    correctCount = 0;

    for i = 1:numInstances

        objectToClassify = data(i,:);
        trueClass = objectToClassify(1);

        nearestNeighborDistance = inf;
        nearestNeighborClass = -1;

        for k = 1:numInstances

            if k == i
                continue;
            end

            distance = 0;

            for j = 1:length(candidateSet)
                featureIndex = candidateSet(j) + 1; % +1 because column 1 is class
                diff = objectToClassify(featureIndex) - data(k, featureIndex);
                distance = distance + diff^2;
            end

            distance = sqrt(distance);

            if distance < nearestNeighborDistance
                nearestNeighborDistance = distance;
                nearestNeighborClass = data(k,1);
            end
        end

        if nearestNeighborClass == trueClass
            correctCount = correctCount + 1;
        end
    end

    accuracy = correctCount / numInstances;
end

function str = setToString(featureSet)
    if isempty(featureSet)
        str = '{}';
        return;
    end

    str = '{';
    for i = 1:length(featureSet)
        str = strcat(str, num2str(featureSet(i)));
        if i < length(featureSet)
            str = strcat(str, ',');
        end
    end
    str = strcat(str, '}');
end