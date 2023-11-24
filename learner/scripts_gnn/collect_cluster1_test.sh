mkdir -p icaps24_test_logs
ssh cluster1 "cd goose-kernels/learner; tar -czvf icaps24_test_logs.tar.gz icaps24_test_logs"
scp cluster1:~/goose-kernels/learner/icaps24_test_logs.tar.gz .
tar -xzvf icaps24_test_logs.tar.gz
rm icaps24_test_logs.tar.gz
