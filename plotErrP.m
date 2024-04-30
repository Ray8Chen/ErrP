% function plotErrP(epochs)
%     % Calculate the mean across epochs
%     meanEpochs = mean(epochs, 3);
% 
%     % Create time vector for plotting
%     timeVector = linspace(-0.2, 0.8, size(meanEpochs, 2));
% 
%     % Plotting
%     figure;
%     hold on;
%     for ch = 1:size(meanEpochs, 1)
%         plot(timeVector, meanEpochs(ch, :));
%         legend(['Channel ', num2str(ch)]);
%     end
%     xlabel('Time (s)');
%     ylabel('Amplitude (uV)');
%     title('Averaged ErrP Waveforms');
%     hold off;
% end

function plotErrP(epochs)
    % Calculate the mean across epochs
    meanEpochs = mean(epochs, 3);  % Averages over the third dimension
    
    % Create time vector for plotting
    % This time vector should be based on the number of time points, which is now the first dimension
    timeVector = linspace(-0.2, 0.8, size(meanEpochs, 1));
    
    % Plotting
    figure;
    hold on;
    for ch = 1:size(meanEpochs, 2)  % Iterate over channels which are now in the second dimension
        plot(timeVector, meanEpochs(:, ch));  % Plotting each channel's mean waveform over time
        legendEntries{ch} = ['Channel ', num2str(ch)];
    end
    legend(legendEntries);  % Create a legend using dynamic entries from the loop
    xlabel('Time (s)');
    ylabel('Amplitude (uV)');
    title('Averaged ErrP Waveforms');
    hold off;
end
