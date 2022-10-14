SRC="https://bop.felk.cvut.cz/media/data/bop_datasets"

wget $SRC/tless_base.zip
wget $SRC/tless_models.zip
wget $SRC/tless_train_pbr.zip
wget $SRC/tless_train_render_reconst.zip
wget $SRC/tless_train_primesense.zip
wget $SRC/tless_test_primesense_all.zip
#wget $SRC/tless_test_primesense_bop19.zip

unzip tless_base.zip             # Contains folder "tless".
unzip tless_models.zip -d tless     # Unpacks to "tless".
unzip tless_train_pbr.zip -d tless  # Unpacks to "tless".
unzip tless_train_render_reconst.zip -d tless   # Unpacks to "tless".
unzip tless_train_primesense.zip -d tless  # Unpacks to "tless".
unzip tless_test_primesense_all.zip -d tless  # Unpacks to "tless".
#unzip tless_test_primesense_bop19.zip -d tless  # Unpacks to "tless".

rm tless_base.zip 
rm tless_models.zip
rm tless_train_pbr.zip
rm tless_train_render_reconst.zip
rm tless_train_primesense.zip
rm tless_test_primesense_all.zip
#rm tless_test_primesense_bop19.zip
