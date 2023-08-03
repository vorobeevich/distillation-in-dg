# OfficeHome from https://github.com/facebookresearch/DomainBed
# create OfficeHome images data

# make directory for all data
mkdir -p data/

# make directory for office_home data
mkdir -p data/office_home/

# load from google drive
echo -e "Start loading images... |"
curl -o data/office_home/office_home_download_file -L "https://drive.google.com/uc?export=download&confirm=yes&id=1uY0pj7oFsjMxRwaD3Sxy0jgel0fsYXLC"

# unpacking download file
echo "Unpacking images... |"
unzip data/office_home/office_home_download_file -d data/office_home/

# remove cache file
rm data/office_home/office_home_download_file