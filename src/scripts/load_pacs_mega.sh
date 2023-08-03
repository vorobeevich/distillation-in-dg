# PACS from https://github.com/liyiying/Feature_Critic
# create pacs images data

# make directory for all data
mkdir -p data/

# make directory for pacs data
mkdir -p data/pacs/

# load images from google drive
echo -e "Start loading images... |"
curl -o data/pacs/pacs_download_file_images -L "https://drive.google.com/uc?export=download&confirm=yes&id=1yDdfWL4Lm5_3-DrqMGSC5dWC-BFYzHq_"

# unpacking download file
echo "Unpacking images... |"
unzip data/pacs/pacs_download_file_images -d data/pacs/

# unzip images to data folder
unzip data/pacs/pacs_data.zip -d data/pacs/

# remove cache files
rm data/pacs/__MACOSX/* -r
rm data/pacs/__MACOSX -r
rm data/pacs/pacs_data/.DS_Store  
rm data/pacs/pacs_download_file_images
rm data/pacs/pacs_data.zip

# rename pacs_data folder to images folder
mv data/pacs/pacs_data data/pacs/images

# create pacs labels data

# make directory for labels data
mkdir -p data/pacs/labels

# load labels from google drive
echo "Start loading labels... |"
curl -o data/pacs/pacs_download_file_labels -L "https://drive.google.com/uc?export=download&confirm=yes&id=1-GZdDFs30abDDT8jmO2agBp0j9tbt_Qy"

# unpacking download file
echo "Unpacking labels... |"

unzip data/pacs/pacs_download_file_labels -d data/pacs/
unzip data/pacs/pacs_label.zip -d data/pacs/labels

# rename and remove files
for domain in art_painting cartoon photo sketch;
  do
    rm data/pacs/labels/${domain}_crossval_kfold.txt
    mv data/pacs/labels/${domain}_train_kfold.txt data/pacs/labels/${domain}_train.txt
    mv data/pacs/labels/${domain}_test_kfold.txt data/pacs/labels/${domain}_test.txt 
  done

# remove cache files
rm data/pacs/pacs_download_file_labels
rm data/pacs/pacs_label.zip
