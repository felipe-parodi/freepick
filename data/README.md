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
1. HVFLIP with FFMPEG
- AVI [MJPEG in WhiteMatter] without hardware acceleration
-- **WINDOWS:** for %f in (*.avi) do ffmpeg -i "%f" -vf "hflip,vflip" -c:v mpeg4 -q:v 1 -c:a copy "flipped\%~nf_flipped.avi"
-- **LINUX:** for f in *.avi; do ffmpeg -i "$f" -vf "hflip,vflip" -c:v mpeg4 -q:v 1 -c:a copy "flipped/${f%.*}_flipped.avi"; done
- MP4 [H264 in WhiteMatter] with h264 hardware acceleration
-- **WINDOWS:** for %f in (*.mp4) do ffmpeg -hwaccel cuda -i "%f" -vf "hflip,vflip" -c:v h264_nvenc -qp 0 -c:a copy "flipped\%~nf_flipped.mp4"
-- **LINUX:** for f in *.mp4; do ffmpeg -hwaccel cuda -i "$f" -vf "hflip,vflip" -c:v h264_nvenc -qp 0 -c:a copy "flipped/${f%.*}_flipped.mp4"; done

2. TRIM TO EXACT FRAME COUNT
- AVI
-- **WINDOWS:** for %i in (*.avi) do (ffmpeg -i "%i" -frames:v FRAME_COUNT -c:v copy -q:v 1 -c:a copy "trim_%i")
-- **LINUX:** for i in *.avi; do ffmpeg -i "$i" -frames:v FRAME_COUNT -c:v copy -q:v 1 -c:a copy "trim_$i"; done
- MP4
-- **WINDOWS:** for %i in (*.mp4) do (ffmpeg -hwaccel cuda -i "%i" -frames:v FRAME_COUNT -c:v h264_nvenc -c:a copy "trim_%i")
-- **LINUX:** for i in *.mp4; do ffmpeg -hwaccel cuda -i "$i" -frames:v FRAME_COUNT -c:v h264_nvenc -c:a copy "trim_$i"; done

3. Ready for model building & analysis!
