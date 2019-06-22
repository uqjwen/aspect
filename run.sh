# python3 model_tf.py -domain laptop -oriented term -train_test train 
# python3 model_tf.py -domain laptop -oriented term -train_test test
# rm -rf ckpt_laptop_term



# python3 model_tf.py -domain laptop -oriented term -train_test train 
# python3 model_tf.py -domain laptop -oriented term -train_test test
# rm -rf ckpt_laptop_term


# python3 model_tf.py -domain laptop -oriented term -train_test train 
# python3 model_tf.py -domain laptop -oriented term -train_test test
# rm -rf ckpt_laptop_term


# python3 model_tf.py -domain laptop -oriented term -train_test train 
# python3 model_tf.py -domain laptop -oriented term -train_test test
# rm -rf ckpt_laptop_term


# python3 model_tf.py -domain laptop -oriented term -train_test train 
# python3 model_tf.py -domain laptop -oriented term -train_test test
# rm -rf ckpt_laptop_term

python3 opinion_extract.py laptop
git add -A 
git commit -m add
git push origin master

python3 opinion_extract.py rest
git add -A
git commit -m add
git push origin master

poweroff