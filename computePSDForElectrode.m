function PSD = computePSDForElectrode(indices, data, Fs, freqRange)
    % Initialize PSD matrix
    PSD = zeros(length(indices), length(freqRange));
    
    % Loop through each event index
    for i = 1:length(indices)
        startIndex = indices(i);
        endIndex = startIndex + Fs - 1;  % One second of data
        
        % Ensure endIndex does not exceed the size of the data
        if endIndex > size(data, 1)
            continue;
        end
        
        % Extract the data segment
        segment = data(startIndex:endIndex);
        
        % Compute PSD
        [pxx, f] = pwelch(segment, [], [], freqRange, Fs);
        
        % Store the PSD values
        PSD(i, :) = pxx;
    end
end
