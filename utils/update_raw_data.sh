# Just a little helper scripts for me that keeps the raw data on NAS up to date

SOURCE=/run/media/sebastian/shoerl_data/population/raw
rsync -avs $SOURCE vplworker@nama:/nas/ivtmatsim/scenario/raw
