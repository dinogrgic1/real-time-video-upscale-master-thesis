$folderPath = "videos\LDV3dataset\validation"

Get-ChildItem -Path $folderPath -Filter *.mkv | ForEach-Object {
    $fileName = $_.Name
    $filePath = $_.FullName
    $fileNumber = $fileName -replace '[^0-9]', ''
    $cropVideoSizes = 128, 256, 384, 512
    foreach ($cropVideoSize in $cropVideoSizes) {
        $newFileName = "{0:D3}.mkv" -f $fileNumber

        $command = "python rt_upscale.py --mode preprocess --preprocess_mode crop_video --original_video_path $filePath --crop_video_size $cropVideoSize"
        Write-Host "Processing file: $fileName"
        Write-Host "Executing command: $command"
        Invoke-Expression $command

        Rename-Item -Path $filePath -NewName $newFileName
    }
}
