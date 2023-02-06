# load pacs images
echo -e "Start loading images... |"
curl -o pacs_download_file_images -L "https://drive.google.com/uc?export=download&confirm=yes&id=1yDdfWL4Lm5_3-DrqMGSC5dWC-BFYzHq_"
echo "Unpacking images... |"
unzip pacs_download_file_images
unzip pacs_data.zip -d ../../data/
rm ../../data/__MACOSX/* -r
rm ../../data/__MACOSX -r
rm ../../data/pacs_data/.DS_Store  
rm pacs_download_file_images
rm pacs_data.zip

mv ../../data/pacs_data ../../data/pacs
mkdir -p ../../data/pacs/images

for folder in art_painting cartoon photo sketch;
  do
    mv ../../data/pacs/${folder}/ ../../data/pacs/images/
  done

# load pacs labels
echo "Start loading labels... |"
curl -o pacs_download_file_labels -L "https://drive.google.com/uc?export=download&confirm=yes&id=1-GZdDFs30abDDT8jmO2agBp0j9tbt_Qy"
echo "Unpacking labels... |"
unzip pacs_download_file_labels
unzip pacs_label.zip

mkdir -p ../../data/pacs/labelsgit 
for domain in art_painting cartoon photo sketch;
  do
    rm ${domain}_crossval_kfold.txt
    mv ${domain}_train_kfold.txt ../../data/pacs/labels/${domain}_train.txt
    mv ${domain}_test_kfold.txt ../../data/pacs/labels/${domain}_test.txt 
  done

rm pacs_download_file_labels
rm pacs_label.zip
