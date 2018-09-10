KALMAN FILTER/ HUNGARIAN ASSIGNMENT VIDEOS:

Videos included:
├── complete_video.mp4
├── bad_example_fast_motion.mp4
├── bad_example_random_switching.mp4
├── good_example_merge_n_separate.mp4
├── good_example_misdetection.mp4
├── good_example_run_n_tumble.mp4
└── good_example_straight_tracks.mp4

Video content:
1. complete_video.mp4: The video output for and entire sequence of 300 frames. 
2. bad_example_fast_motion.mp4: The purple track loses its fast moving cell, and cell generates new red track.
3. bad_example_random_switching.mp4: The 2 purple tracks merge, and tracks switch randomly for merged cells.
4. good_example_merge_n_separate.mp4: Pink and blue tracks maintain velocity from motion model, and keep tracking corresponding cells without switching paths.
5. good_example_misdetection.mp4: Blue track in bottom left tracks detection well, even when cell is not segmented by DeepCell.
6. good_example_run_n_tumble.mp4: Blue track tracks its cell well through the entirety of its tumble phase.
7. good_example_straight_tracks.mp4: Several good tracks for straight cell trajectories are observed.