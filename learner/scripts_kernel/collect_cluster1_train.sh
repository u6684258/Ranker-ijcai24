mkdir -p icaps24_train_logs
mkdir -p icaps24_wl_models

ssh cluster1 "cd goose-kernels/learner; tar -czvf icaps24_train_logs.tar.gz icaps24_train_logs; tar -czvf icaps24_wl_models.tar.gz icaps24_wl_models"

scp cluster1:~/goose-kernels/learner/icaps24_train_logs.tar.gz .
scp cluster1:~/goose-kernels/learner/icaps24_wl_models.tar.gz .

tar -xzvf icaps24_train_logs.tar.gz
tar -xzvf icaps24_wl_models.tar.gz

rm icaps24_train_logs.tar.gz
rm icaps24_wl_models.tar.gz
