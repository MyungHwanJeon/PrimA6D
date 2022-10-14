SRC="https://bop.felk.cvut.cz/media/data/bop_datasets"

wget $SRC/ycbv_base.zip
wget $SRC/ycbv_models.zip
wget $SRC/ycbv_train_pbr.zip
wget $SRC/ycbv_train_synt.zip
wget $SRC/ycbv_train_real.zip
wget $SRC/ycbv_test_all.zip
#wget $SRC/ycbv_test_bop19.zip

unzip ycbv_base.zip             # Contains folder "ycbv".
unzip ycbv_models.zip -d ycbv     # Unpacks to "ycbv".
unzip ycbv_train_pbr.zip -d ycbv # Unpacks to "ycbv".
unzip ycbv_train_synt.zip -d ycbv   # Unpacks to "ycbv".
unzip ycbv_train_real.zip -d ycbv  # Unpacks to "ycbv".
unzip ycbv_test_all.zip -d ycbv  # Unpacks to "ycbv".
#unzip ycbv_test_bop19.zip -d ycbv  # Unpacks to "ycbv".

rm ycbv_base.zip  
rm ycbv_models.zip
rm ycbv_train_pbr.zip
rm ycbv_train_synt.zip
rm ycbv_train_real.zip
rm ycbv_test_all.zip 
#rm ycbv_test_bop19.zip
