function epochs = extractEpochs(eegData, eventMarkers, preEventDuration, postEventDuration, samplingRate)
    % Calculate the number of samples before and after the event
    preEventSamples = round(preEventDuration * samplingRate);
    postEventSamples = round(postEventDuration * samplingRate);
    
    % Initialize the epochs matrix
    % Number of channels is the number of columns in eegData
    numChannels = size(eegData, 2);
    % Number of epochs is the length of eventMarkers
    numEpochs = length(eventMarkers);
    % Each epoch length in samples
    epochLength = preEventSamples + postEventSamples + 1;
    
    % Initialize the 3D matrix to hold the epochs
    epochs = zeros(epochLength, numChannels, numEpochs);
    
    % Extract epochs
    for i = 1:numEpochs
        % Calculate the start and end indices for the current epoch
        startIndex = eventMarkers(i) - preEventSamples;
        endIndex = eventMarkers(i) + postEventSamples;
        
        % Check if the calculated indices are within the bounds of eegData
        if startIndex < 1 || endIndex > size(eegData, 1)
            warning('Epoch %d is partially or fully outside of data range and will be skipped.', i);
            continue;
        end
        
        % Extract the epoch and store it in the matrix
        epochs(:, :, i) = eegData(startIndex:endIndex, :);
    end
end
