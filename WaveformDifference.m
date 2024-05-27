% Define the path to your CSV file
filePath = 'Jianan-testing-[2024.05.16-15.18.15].csv';  % Update this with the correct path

% Load the data
data = readtable(filePath);

% Convert table to array 
eegData = table2array(data);
% eegData = eegData(3:18);

% Replace NaN values in 'Event Id' with a placeholder 
% if any(ismissing(eegData.EventId))
%     eegData.EventId = fillmissing(eegData.EventId, 'constant', -1);  % Replace NaN with -1
% end

% Replace NaN values with zeros
eegData(isnan(eegData)) = -1;


% Extract EEG channel data and Event Id
eegChannels = eegData(:, 3:18);  % Adjust the indices based on specific columns
eventIds = eegData(:, 19);


% Assume eegData is the data matrix, samplingRate is 512 Hz
% and eventIds is the vector of Event Ids

% Step 1: Filter the data
filteredEEG = filteredData(eegChannels, 512);

% Step 2: Extract error and correct epochs
errorEpochs = errorepochs(filteredEEG, eventIds, 512);
correctEpochs = correctepochs(filteredEEG, eventIds, 512);

% Step 3: Plot ErrP
plotErrPCorrectDifference(correctEpochs, errorEpochs, 512);