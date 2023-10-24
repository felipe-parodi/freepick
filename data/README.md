# Open-Field Enclosure Foraging

## Dataset
| Session  | NumOfCams | Length (min:sec) | FPS | Resolution | FrameCt | Processing     | Type        | Ready | Note                                                                         |
|----------|-----------|------------------|-----|------------|---------|----------------|-------------|-------|------------------------------------------------------------------------------|
| 08302022 | 30        | 14:39            | 30  | 1920x1080  | 26400   | v,hflip        | Mat & more  | No    | No foraging and cuts off before he leaves? All same length except 8239.       |
| 08312022 | 30        | 12:15            | 30  | 1920x1080  | 22080   | v,hflip        | Mat & more  | No    | Forages a bit. Then goes up. Then comes down and forages.                    |
| 09012022 | 30        | 05:31            | 30  | 1920x1080  | 9960    | v,hflip        | Mat         | No    | Like prior session                                                            |
| 09062022 | 29        | 18:08            | 15  | 640x480    | 16320   | v,hflip        | Mat         | No    | Like prior session                                                            |
| 09072022 | 29        | 19:00            | 30  | 1920x1080  |         | v,hflip,sync   | Mat         | No    | All videos unsynced                                                           |
| 20221013 | 29        | 02:40            | 30  | 1920x1080  |         | v,hflip,sync   | Paint Roller| No    |                                                                              |
| 20221018 | 29        | First            | 30  | 1920x1080  |         | v,hflip,sync   | Paint Roller| No    | Smaller videos seem to be the foraging part. Deleted the heavier videos… but the main videos are all corrupted :/|
| 20221019 | 28        | First            | 30  | 1920x1080  |         | v,hflip,sync   | Paint Roller| No    | Larger videos seem to be the foraging part. Not corrupt!                     |
| 20221020 | 28        | First            | 30  | 1920x1080  |         | v,hflip,sync   | Paint Roller| No    |                                                                              |
| Total duration |       |                  |     |            |         |                |             |       |                                                                              |


## Preprocessing steps

### Flip dir of videos with ffmpeg
- Linux: for f in *.avi; do ffmpeg -hwaccel cuda -i "$f" -vf "hflip,vflip" -c:v h264_nvenc -c:a copy "flipped/${f%.avi}_flipped.avi"; done
- Windows: for %f in (*.avi) do ffmpeg -hwaccel cuda -i "%f" -vf "hflip,vflip" -c:v h264_nvenc -c:a copy "flipped\%~nf_flipped.avi"
- For one video: ffmpeg -hwaccel cuda -i input.avi -vf "hflip,vflip" -c:v h264_nvenc -c:a copy output.avi

### Trim to exact frame count
- ...
- 
