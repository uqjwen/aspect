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


rm -rf ckpt_laptop_term_50_128
rm -rf ckpt_laptop_term_200_128
rm -rf ckpt_laptop_term_300_128

rm -rf ckpt_laptop_term_100_16
rm -rf ckpt_laptop_term_100_32
rm -rf ckpt_laptop_term_100_64
# rm -rf ckpt_laptop_term_100_128


rm -rf ckpt_rest_term_50_128

rm -rf ckpt_rest_term_200_128
rm -rf ckpt_rest_term_300_128

rm -rf ckpt_rest_term_100_16
rm -rf ckpt_rest_term_100_32
rm -rf ckpt_rest_term_100_64





python3 model_tf.py -domain laptop -oriented term -emb_size 100 -train_test train 
# python3 model_tf.py -domain laptop -oriented term -emb_size 200 -train_test train 
# python3 model_tf.py -domain laptop -oriented term -emb_size 300 -train_test train 

# python3 model_tf.py -domain laptop -oriented term -filter_map 32 -train_test train 
# python3 model_tf.py -domain laptop -oriented term -filter_map 64 -train_test train 
# python3 model_tf.py -domain laptop -oriented term -filter_map 16 -train_test train 



python3 model_tf.py -domain rest -oriented term -emb_size 100 -train_test train 
# python3 model_tf.py -domain rest -oriented term -emb_size 200 -train_test train 
# python3 model_tf.py -domain rest -oriented term -emb_size 300 -train_test train 

rm -rf ckpt_laptop_term_100_128

rm -rf ckpt_rest_term_100_128



# python3 model_tf.py -domain rest -oriented term -filter_map 32 -train_test train 
# python3 model_tf.py -domain rest -oriented term -filter_map 64 -train_test train 
# python3 model_tf.py -domain rest -oriented term -filter_map 16 -train_test train 

git add -A
git commit -m changes
git push origin master
poweroff