mkdir -p icaps24_wl_models
cd icaps24_wl_models
rm -f zip.zip
zip -r zip.zip *
scp zip.zip gadi:/scratch/xb83/dc6693/goose-kernels/learner/icaps24_wl_models

