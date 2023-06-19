folderPath = 'Test_data/test6/';
fileList = dir(fullfile(folderPath, 'img*.png'));

% Sort the file list based on frame numbers
frameNumbers = cellfun(@(x) str2double(x(4:end-4)), {fileList.name});
[~, sortedIndices] = sort(frameNumbers);
fileList = fileList(sortedIndices);

outputVideo = VideoWriter('timelapse_video.mp4', 'MPEG-4');
outputVideo.FrameRate = 30;
open(outputVideo);

for i = 1:numel(fileList)
    filePath = fullfile(folderPath, fileList(i).name);
    img = imread(filePath);
    writeVideo(outputVideo, img);
end

close(outputVideo);