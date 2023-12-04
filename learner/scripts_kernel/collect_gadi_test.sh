mkdir -p icaps24_test_logs

# echo "collecting combined logs..."
# ssh gadi "cd /scratch/xb83/dc6693/goose-kernels/learner/; zip -r icaps24_test_logs.zip icaps24_test_logs/*combined*"

# echo "collecting gpr logs..."
# ssh gadi "cd /scratch/xb83/dc6693/goose-kernels/learner/; zip -r icaps24_test_logs.zip icaps24_test_logs/*gp*"

# echo "collecting combined gpr logs..."
# ssh gadi "cd /scratch/xb83/dc6693/goose-kernels/learner/; zip -r icaps24_test_logs.zip icaps24_test_logs/*combined_gp*"

echo "collecting MIP logs..."
ssh gadi "cd /scratch/xb83/dc6693/goose-kernels/learner/; zip -r icaps24_test_logs.zip icaps24_test_logs/*mip*"

scp gadi:/scratch/xb83/dc6693/goose-kernels/learner/icaps24_test_logs.zip .

unzip -o icaps24_test_logs.zip
