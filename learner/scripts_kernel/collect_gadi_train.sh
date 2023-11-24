mkdir -p icaps24_train_logs
ssh gadi "cd /scratch/xb83/dc6693/goose-kernels/learner/; zip -r icaps24_train_logs.zip icaps24_train_logs; zip -r icaps24_wl_models.zip icaps24_wl_models"
scp gadi:/scratch/xb83/dc6693/goose-kernels/learner/icaps24_train_logs.zip .
scp gadi:/scratch/xb83/dc6693/goose-kernels/learner/icaps24_wl_models.zip .
unzip icaps24_train_logs.zip
unzip icaps24_wl_models.zip
rm icaps24_train_logs.zip
rm icaps24_wl_models.zip
