SRC="https://bop.felk.cvut.cz/media/data/bop_datasets"

wget $SRC/lm_base.zip
wget $SRC/lm_models.zip
wget $SRC/lm_test_bop_19.zip
wget $SRC/lm_test_all.zip
wget $SRC/lm_train.zip
wget $SRC/lm_train_pbr.zip

unzip lm_base.zip             # Contains folder "lm".
unzip lm_models.zip -d lm     # Unpacks to "lm".
#unzip lm_test_bop_19.zip -d lm  # Unpacks to "lm".
unzip lm_test_all.zip -d lm   # Unpacks to "lm".
unzip lm_train.zip -d lm  # Unpacks to "lm".
unzip lm_train_pbr.zip -d lm  # Unpacks to "lm".

rm lm_base.zip 
rm lm_models.zip
#rm lm_test_bop_19.zip
rm lm_test_all.zip
rm lm_train_pbr.zip
rm lm_train.zip
