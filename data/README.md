# Open-Field Enclosure Foraging

## Dataset
| Session  | NumOfCams | Length (min:sec) | FPS | Resolution | FrameCt | Needs | Stage | Type        | Ready | Note                                                                         |
|----------|-----------|------------------|-----|------------|---------|----------------|---|------------|-------|-----|
| 08302022 | 30        | 14:39            | 30  | 1920x1080  | 26400   | v,hflip        | Needs sync | Mat & more  | No    | N/A |
| 08312022 | 30        | 12:15            | 30  | 1920x1080  | 22080   | v,hflip        | Needs sync | Mat & more  | No    | N/A |
| 09012022 | 30        | 05:31            | 30  | 1920x1080  | 9960    | v,hflip        | Needs sync | Mat         | No    | N/A |
| 09062022 | 29        | 18:08            | 15  | 640x480    | 16320   | v,hflip        | Needs sync | Mat         | No    | N/A |
| 09072022 | 29        | 19:00            | 30  | 1920x1080  |         | v,hflip,sync   | Needs sync | Mat         | No    | N/A |
| 20221013 | 29        | 02:40            | 30  | 1920x1080  |         | v,hflip,sync   | Needs sync | Paint Roller| No    | N/A |
| 20221018 | 29        | First            | 30  | 1920x1080  |         | v,hflip,sync   | Needs sync | Paint Roller| No    | Corrupted |
| 20221019 | 28        | First            | 30  | 1920x1080  |         | v,hflip,sync   | Needs sync | Paint Roller| No    | N/A |
| 20221020 | 28        | First            | 30  | 1920x1080  |         | v,hflip,sync   | Needs sync | Paint Roller| No    | N/A |
| Total duration |       |                  |     |            |         |                |         |     |       |        |


## Preprocessing Stages

1. Flip videos with ffmpeg.
- Linux directory: for f in *.avi; do ffmpeg -hwaccel cuda -i "$f" -vf "hflip,vflip" -c:v h264_nvenc -c:a copy "flipped/${f%.avi}_flipped.avi"; done
- Windows directory: for %f in (*.avi) do ffmpeg -hwaccel cuda -i "%f" -vf "hflip,vflip" -c:v h264_nvenc -c:a copy "flipped\%~nf_flipped.avi"
- Linux single video: ffmpeg -hwaccel cuda -i input.avi -vf "hflip,vflip" -c:v h264_nvenc -c:a copy output.avi

2. Select top 3 angles.

3. Decide on video length and trim to exact frame count (AKA sync).
- ...

4. Model building. 
