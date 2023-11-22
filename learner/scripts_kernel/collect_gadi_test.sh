mkdir -p icaps24_logs
ssh gadi "cd /scratch/xb83/dc6693/goose-kernels/learner/; zip -r icaps24_logs.zip icaps24_logs"
scp gadi:/scratch/xb83/dc6693/goose-kernels/learner/icaps24_logs.zip .
unzip icaps24_logs.zip
