# OfficeHome from https://github.com/facebookresearch/DomainBed
# create OfficeHome images data

# make directory for all data
mkdir -p data/

# make directory for officehome data
mkdir -p data/officehome/

# load from google drive
echo -e "Start loading images... |"
curl -o data/officehome/officehome_download_file -L "https://drive.google.com/uc?export=download&confirm=yes&id=1uY0pj7oFsjMxRwaD3Sxy0jgel0fsYXLC"

# unpacking download file
echo "Unpacking images... |"
unzip data/officehome/officehome_download_file -d data/officehome/

# remove cache file
rm data/officehome/officehome_download_file

# rename OfficeHomeDataset_10072016 folder to images folder
mv data/officehome/OfficeHomeDataset_10072016/ data/officehome/images

# remove space from domain name
mv "data/officehome/images/Real World/" data/officehome/images/Real_World

# remove cache files
rm data/officehome/images/ImageInfo.csv
rm data/officehome/images/imagelist.txt

# make directory for labels data
mkdir -p data/officehome/labels/