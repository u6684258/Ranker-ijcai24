mkdir -p icaps24_test_logs

ssh gadi "cd /scratch/xb83/dc6693/goose-kernels/learner/; zip -r icaps24_test_logs.zip icaps24_test_logs"

scp gadi:/scratch/xb83/dc6693/goose-kernels/learner/icaps24_test_logs.zip .

unzip -o icaps24_test_logs.zip
