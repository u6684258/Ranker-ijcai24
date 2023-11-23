mkdir -p icaps24_train_logs
ssh gadi "cd /scratch/xb83/dc6693/goose-kernels/learner/; zip -r icaps24_train_logs.zip icaps24_train_logs"
scp gadi:/scratch/xb83/dc6693/goose-kernels/learner/icaps24_train_logs.zip .
unzip icaps24_train_logs.zip
rm icaps24_train_logs.zip
