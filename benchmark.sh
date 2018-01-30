#!/bin/bash

runs=$1
iter=$2
size=$3
arch=$4
stencil=$5

echo "radius" | xargs printf "%-10s"
echo "time" | xargs printf "%-15s"
echo "points" | xargs printf "%-15s"
echo "flops" | xargs printf "%-15s"
echo "power" | xargs printf "%-15s"
echo

for i in {1..4}
do
	timesum=0
	pointsum=0
	flopssum=0
	powersum=0
	make clean >/dev/null 2>&1; make stencil=$stencil arch=$arch radius=$i -j 6 >/dev/null 2>&1
	if [ "$arch" == "knl" ]
	then
		out=`sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH:lib/ numactl --preferred=1 bin/yask_kernel.$stencil.$arch.exe -d $size -t $runs -dt $iter`
	else
		out=`sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH:lib/ numactl --cpunodebind=0 bin/yask_kernel.$stencil.$arch.exe -d $size -t $runs -dt $iter`
	fi
	for (( k=1; k<=$runs; k++ ))
	do
		time=`echo "$out" | grep elapsed-time -m1 | tail -n1 | cut -d ":" -f 2 | tr -d '[:space:]'`
		temp=`echo "$out" | grep throughput | grep "num-points" -m $k | tail -n 1 | cut -d ":" -f 2 | tr -d '[:space:]' | numfmt --from=auto`
		point=`echo "scale=3; $temp/1000000000.0" | bc -l`
		temp=`echo "$out" | grep FLOPS -m $k | tail -n 1 | cut -d ":" -f 2 | tr -d '[:space:]' | numfmt --from=auto`
		flops=`echo "scale=3; $temp/1000000000.0" | bc -l`
		power=`echo "$out" | grep average-power -m1 | tail -n1 | cut -d ":" -f 2 | tr -d '[:space:]'`
		timesum=`echo $pointsum+$point | bc -l`
		pointsum=`echo $pointsum+$point | bc -l`
		flopssum=`echo $flopssum+$flops | bc -l`
		powersum=`echo $powersum+$power | bc -l`
	done
	pointaverage=`echo $pointsum/$runs | bc -l`
	flopsaverage=`echo $flopssum/$runs | bc -l`
	poweraverage=`echo $powersum/$runs | bc -l`
	timeaverage=`echo $powersum/$runs | bc -l`
	echo $i | xargs printf "%-10s"
	echo $timeaverage | xargs printf "%-15.3f"
	echo $pointaverage | xargs printf "%-15.3f"
	echo $flopsaverage | xargs printf "%-15.3f"
	echo $poweraverage | xargs printf "%-15.3f"
	echo
done
