$folderPath = "videos\LDV3dataset\"

# Loop through all files in the folder
Get-ChildItem -Path $folderPath -Filter *.mkv | ForEach-Object {
    $fileName = $_.Name
    $filePath = $_.FullName

    # Extract the file number from the file name
    $fileNumber = $fileName -replace '[^0-9]', ''

    # Iterate through different crop video sizes
    $cropVideoSizes = 128, 256, 384, 512
    foreach ($cropVideoSize in $cropVideoSizes) {
        # Create the new file name with the updated crop video size
        $newFileName = "{0:D3}.mkv" -f $fileNumber
        $newFilePath = Join-Path -Path $folderPath -ChildPath $newFileName

        # Modify the crop video size in the command and execute it
        $command = "python rt_upscale.py --mode preprocess --preprocess_mode crop_video --original_video_path $filePath --crop_video_size $cropVideoSize"
        Write-Host "Processing file: $fileName"
        Write-Host "Executing command: $command"
        Invoke-Expression $command

        # Rename the output file to the new file name
        Rename-Item -Path $filePath -NewName $newFileName
    }
}
