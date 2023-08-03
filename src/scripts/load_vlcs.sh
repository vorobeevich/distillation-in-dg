# VLCS from https://github.com/facebookresearch/DomainBed
# create vlcs images data

# make directory for all data
mkdir -p data/

# make directory for vlcs data
mkdir -p data/vlcs/

# load from google drive
echo -e "Start loading images... |"
curl -o data/vlcs/vlcs_download_file -L "https://drive.google.com/uc?export=download&confirm=yes&id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8"

# unpacking download file
echo "Unpacking images... |"
unzip data/vlcs/vlcs_download_file -d  data/vlcs/

# remove cache file
rm data/vlcs/vlcs_download_file

# rename VLCS folder to images folder
mv data/vlcs/VLCS/ data/vlcs/images

# make directory for labels data
mkdir -p data/vlcs/labels/