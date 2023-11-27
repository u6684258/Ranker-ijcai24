mkdir -p icaps24_gp_correlation_logs

ssh gadi "cd /scratch/xb83/dc6693/goose-kernels/learner/; zip -r icaps24_gp_correlation_logs.zip icaps24_gp_correlation_logs"

scp gadi:/scratch/xb83/dc6693/goose-kernels/learner/icaps24_gp_correlation_logs.zip .

unzip -o icaps24_gp_correlation_logs.zip
