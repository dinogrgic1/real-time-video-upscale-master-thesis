$folderPaths = "videos\LDV3dataset\test\cropped_"
$cropped_sizes = 512,384,256,128

foreach ($folderPath in $folderPaths) {
    foreach ($cropped_size in $cropped_sizes)
    {
Get-ChildItem -Path $folderPath$cropped_size -Filter *.mkv | ForEach-Object {
    $fileName = $_.Name
    $filePath = $_.FullName
    $cropVideoSizes = 0.25, 0.5
    foreach ($cropVideoSize in $cropVideoSizes)
    {
        $command = "python rt_upscale.py --mode preprocess --preprocess_mode downscale_video --original_video_path $filePath --downscale_video_ratio $cropVideoSize"
        Write-Host "Processing file: $fileName"
        Write-Host "Executing command: $command"
        Invoke-Expression $command
    }
}
    }
}
