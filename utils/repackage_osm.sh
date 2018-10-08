SOURCE=/path/to/switzerland-latest.osm.bz2
TARGET=/path/to/switzerland-latest.osm.gz

bunzip2 -c $SOURCE | gzip -c > $TARGET
