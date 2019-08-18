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

# python3 opinion_extract.py laptop
# git add -A 
# git commit -m add
# git push origin master

# python3 opinion_extract.py rest
# git add -A
# git commit -m add
# git push origin master


# python3 emb_mat.py laptop 50
# python3 emb_mat.py laptop 200
# python3 emb_mat.py laptop 300

# python3 emb_mat.py rest 50
# python3 emb_mat.py rest 200
# python3 emb_mat.py rest 300


python3 model_tf.py -domain laptop -oriented term -emb_size 50 -train_test train 
python3 model_tf.py -domain laptop -oriented term -emb_size 50 -train_test test 

python3 model_tf.py -domain laptop -oriented term -emb_size 200 -train_test train 
python3 model_tf.py -domain laptop -oriented term -emb_size 200 -train_test test 

python3 model_tf.py -domain laptop -oriented term -emb_size 300 -train_test train 
python3 model_tf.py -domain laptop -oriented term -emb_size 300 -train_test test 



# poweroff