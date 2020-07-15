

set -x

source ~/scripts/sourceme.sh gperftools
source ~/scripts/sourceme.sh hwloc
source ~/scripts/sourceme.sh vc
source ~/scripts/sourceme.sh silo
source ~/scripts/sourceme.sh $1/hpx

rm -rf $1
mkdir $1
cd $1
rm CMakeCache.txt
rm -r CMakeFiles


cmake -DCMAKE_PREFIX_PATH="$HOME/local/$1/hpx" \
      -DCMAKE_CXX_COMPILER=icpc  \
      -DCMAKE_C_COMPILER=icc \
      -DCMAKE_CXX_FLAGS="" \
      -DCMAKE_C_FLAGS="" \
      -DCMAKE_BUILD_TYPE=$1                                                                                                                            \
      -DBOOST_ROOT=$HOME/local/boost \
      -DHDF5_ROOT=$HOME/local/hdf5 \
      -DSilo_DIR=$HOME/local/silo \
      ..


make -j VERBOSE=1
make test VERBOSE=1


