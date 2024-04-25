% Define the path to your CSV file
filePath = 'Ziyue-testing-record-[2024.04.18-17.14.08] (1).csv';  % Update this with the correct path

% Load the data
data = readtable(filePath);

% Convert table to array 
eegData = table2array(data);


% Replace NaN values in 'Event Id' with a placeholder 
% if any(ismissing(eegData.EventId))
%     eegData.EventId = fillmissing(eegData.EventId, 'constant', -1);  % Replace NaN with -1
% end

% Replace NaN values with zeros
eegData(isnan(eegData)) = -1;


% Extract EEG channel data and Event Id
eegChannels = eegData{:, 3:18};  % Adjust the indices based on your specific columns
eventIds = eegData.EventId;


% Assume eegData is the data matrix, samplingRate is 512 Hz
% and eventIds is the vector of Event Ids

% Step 1: Filter the data
filteredEEG = applyBandPassFilter(eegData, 512);

% Step 2: Extract error epochs
epochs = extractErrorEpochs(filteredEEG, eventIds, 512);

% Step 3: Plot ErrP
plotErrP(epochs);


function filteredData = applyBandPassFilter(data, samplingRate)
    % Design the band-pass filter
    lowCutoff = 1;  % Low frequency cutoff (Hz)
    highCutoff = 10;  % High frequency cutoff (Hz)
    [b, a] = butter(4, [lowCutoff highCutoff] / (samplingRate / 2), 'bandpass');
    
    % Apply the filter
    filteredData = filtfilt(b, a, data);
end

function epochs = extractErrorEpochs(filteredData, eventIds, samplingRate)
    errorEventIndices = find(eventIds == 0);
    epochWindow = [-0.2, 0.8];  % Seconds before and after the event
    samplesWindow = round(epochWindow * samplingRate);
    epochLength = range(samplesWindow) + 1;
    epochs = zeros(size(filteredData, 1), epochLength, numel(errorEventIndices));
    
    for i = 1:numel(errorEventIndices)
        startIndex = max(1, errorEventIndices(i) + samplesWindow(1));
        endIndex = min(size(filteredData, 2), errorEventIndices(i) + samplesWindow(2));
        epochs(:, :, i) = filteredData(:, startIndex:endIndex);
    end
end

function plotErrP(epochs)
    % Calculate the mean across epochs
    meanEpochs = mean(epochs, 3);
    
    % Create time vector for plotting
    timeVector = linspace(-0.2, 0.8, size(meanEpochs, 2));
    
    % Plotting
    figure;
    hold on;
    for ch = 1:size(meanEpochs, 1)
        plot(timeVector, meanEpochs(ch, :));
        legend(['Channel ', num2str(ch)]);
    end
    xlabel('Time (s)');
    ylabel('Amplitude (uV)');
    title('Averaged ErrP Waveforms');
    hold off;
end

