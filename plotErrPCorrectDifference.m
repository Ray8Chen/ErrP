function plotErrPCorrectDifference(correctEpochs, errorEpochs, samplingRate)
    % Validate input dimensions
    if size(correctEpochs, 2) ~= size(errorEpochs, 2) || ...
       size(correctEpochs, 1) ~= size(errorEpochs, 1)
        error('Input dimensions of correctEpochs and errorEpochs must match.');
    end
    
    % Calculate the mean across events for both conditions
    meanCorrect = mean(correctEpochs, 3);
    meanError = mean(errorEpochs, 3);
    
    % Calculate the difference between error and correct mean waveforms
    diffWaveforms = meanError - meanCorrect;
    
    % Create time vector for plotting, assuming epoch windows are the same
    epochWindow = [-0.2, 0.8];  % Adjust based on your actual epoch window
    timeVector = linspace(epochWindow(1), epochWindow(2), size(meanCorrect, 1));
    
    % Plotting
    figure;
    hold on;
    % Plot the difference for each channel
    for ch = 1:size(diffWaveforms, 2)
        plot(timeVector, diffWaveforms(:, ch), 'DisplayName', sprintf('Channel %d', ch));
    end
    hold off;
    
    % Customize the plot
    legend show;
    xlabel('Time (s)');
    ylabel('Amplitude Difference (uV)');
    title('Difference in Averaged Waveforms (Error - Correct)');
    grid on;
end
