$folderPaths = "videos\LDV3dataset\validation\cropped_"
$gtFolderPaths = "videos\LDV3dataset\validation\005.mkv"
$cropped_sizes = 512

foreach ($folderPath in $folderPaths) {
    foreach ($cropped_size in $cropped_sizes)
    {
        Get-ChildItem -Path $folderPath$cropped_size -Filter *.mkv | ForEach-Object {
            $fileName = $_.Name
            $filePath = $_.FullName
            $command = "python rt_upscale.py --mode upscale --onnx_engine_path models/saved/onnx/model_070623_191018_1x512x512x3.engine --original_video_path $filePath --gt_video_path $gtFolderPaths"
            Write-Host "Processing file: $fileName"
            Write-Host "Executing command: $command"
            Invoke-Expression $command
        }
    }
}
