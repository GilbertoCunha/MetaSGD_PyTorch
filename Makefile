run: miniimagenet_train.py learner.py meta.py
	@python miniimagenet_train.py --k_spt 5 --vvs_depth 6 --update_step 0 --update_step_test 0
	@python miniimagenet_train.py --k_spt 5 --vvs_depth 7 --update_step 0 --update_step_test 0
	@python miniimagenet_train.py --k_spt 5 --vvs_depth 8 --update_step 0 --update_step_test 0