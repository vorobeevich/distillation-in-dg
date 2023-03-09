# create pacs images data

# make directory for all data
mkdir -p data/

# make directory for pacs data
mkdir -p data/pacs/

# load from google drive
echo -e "Start loading images... |"
curl -o data/pacs/pacs_download_file -L "https://drive.google.com/uc?export=download&confirm=yes&id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd"

# unpacking download file
echo "Unpacking images... |"
unzip data/pacs/pacs_download_file -d data/pacs/

# remove cache file
rm data/pacs/pacs_download_file

# rename kfold folder to images folder
mv data/pacs/kfold/ data/pacs/images

# make directory for labels data
mkdir -p data/pacs/labels