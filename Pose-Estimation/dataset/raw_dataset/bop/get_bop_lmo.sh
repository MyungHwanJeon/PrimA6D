SRC="https://bop.felk.cvut.cz/media/data/bop_datasets"

wget $SRC/lmo_base.zip
wget $SRC/lmo_models.zip
wget $SRC/lmo_train_pbr.zip
wget $SRC/lmo_train.zip
wget $SRC/lmo_test_all.zip
#wget $SRC/lmo_test_bop_19.zip

unzip lmo_base.zip             # Contains folder "lmo".
unzip lmo_models.zip -d lmo     # Unpacks to "lmo".
unzip lmo_train_pbr.zip -d lmo  # Unpacks to "lmo".
unzip lmo_train.zip -d lmo   # Unpacks to "lmo".
unzip lmo_test_all.zip -d lmo  # Unpacks to "lmo".
#unzip lmo_test_bop_19.zip -d lmo  # Unpacks to "lmo".

rm lmo_base.zip 
rm lmo_models.zip
rm lmo_train_pbr.zip
rm lmo_train.zip
rm lmo_test_all.zip
#rm lmo_test_bop_19.zip
